import random
from abc import ABC, abstractmethod
from itertools import chain, zip_longest
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdMMPA import FragmentMol
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

import flowr.util.functional as smolF
from flowr.util.ifg import identify_functional_groups
from flowr.util.molrepr import GeometricMol, GeometricMolBatch, SmolBatch, SmolMol
from flowr.util.pocket import PocketComplex, ProteinPocket

PLINDER_MOLECULE_SIZE_MEAN = 48.3841740914044
PLINDER_MOLECULE_SIZE_STD_DEV = 20.328270251327584
PLINDER_MOLECULE_SIZE_MAX = 182
PLINDER_MOLECULE_SIZE_MIN = 8
CROSSDOCKED_MOLECULE_SIZE_MEAN = 40.0
CROSSDOCKED_MOLECULE_SIZE_STD_DEV = 10.0
CROSSDOCKED_MOLECULE_SIZE_MAX = 82
CROSSDOCKED_MOLECULE_SIZE_MIN = 5
KINODATA_MOLECULE_SIZE_MEAN = 31.24166706404082
KINODATA_MOLECULE_SIZE_STD_DEV = 6.369577265037612
KINODATA_MOLECULE_SIZE_MAX = 84
KINODATA_MOLECULE_SIZE_MIN = 4

_InterpT = tuple[list[SmolMol], list[SmolMol], list[SmolMol], list[torch.Tensor]]
_GeometricInterpT = tuple[
    list[GeometricMol], list[GeometricMol], list[GeometricMol], list[torch.Tensor]
]
_ComplexInterpT = tuple[
    list[PocketComplex], list[PocketComplex], list[PocketComplex], list[torch.Tensor]
]


def extract_fragments(to_mols: list[Chem.Mol], maxCuts: int = 3):
    def fragment_per_mol(mol: Chem.Mol, maxCuts: int):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Fragments could not be extracted as molecule could not be sanitized. Skipping!"
            )
            return mask

        def clean_fragment(frag):
            for a in frag.GetAtoms():
                if a.GetAtomicNum() == 0:
                    a.SetAtomicNum(1)
            frag = Chem.RemoveHs(frag)
            frags = Chem.GetMolFrags(frag, asMols=True)
            return frags

        # Generate fragments
        frags = FragmentMol(mol=mol, maxCuts=maxCuts)
        frags = [clean_fragment(frag) for frag in chain(*frags) if frag]
        frags = [
            frag
            for frag_tuple in frags
            for frag in frag_tuple
            if frag.GetNumAtoms() > 1
        ]
        substructure_ids = [mol.GetSubstructMatches(frag)[0] for frag in frags]
        # Randomly select a fragment
        findices = []
        if substructure_ids:
            frag = random.choice(substructure_ids)
            findices.extend(frag)

            if findices:
                mask[torch.tensor(findices)] = 1
        return mask

    return [fragment_per_mol(mol) for mol in to_mols]


def extract_substructure(
    to_mols: list[Chem.Mol], substructure_query: str, use_smarts: bool = False
):
    def substructure_per_mol(mol, substructure_query):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Substructure could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        if use_smarts:
            substructure = Chem.MolFromSmarts(substructure_query)
        else:
            substructure = Chem.MolFromSmiles(substructure_query)
        if substructure is None or substructure.GetNumAtoms() == 0:
            print(
                "Substructure could not be extracted from the reference molecule. Skipping."
            )
            return mask
        if mol.HasSubstructMatch(
            substructure
        ):  # TODO: handle the case where multiple substructures are present
            try:
                substructure_atoms = mol.GetSubstructMatches(substructure)
            except Exception as e:
                print(e)
        if len(substructure_atoms) > 0:
            mask[torch.tensor(substructure_atoms)] = 1
        return mask

    def substructure_per_mol_list(mol, substructure_atoms):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Substructure could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        mask[torch.tensor(substructure_atoms)] = 1
        return mask

    if isinstance(substructure_query, str):
        mask = [substructure_per_mol(mol, substructure_query) for mol in to_mols]
    elif isinstance(substructure_query, list):
        mask = [substructure_per_mol_list(mol, substructure_query) for mol in to_mols]
    else:
        raise ValueError("substructure_query must be a string or a list of strings.")

    return mask


def extract_func_groups(to_mols: list[Chem.Mol], includeHs=False):
    def func_groups_per_mol(mol, includeHs=True):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Functional groups could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        fgroups = identify_functional_groups(mol)
        findices = []
        for f in fgroups:
            findices.extend(f.atomIds)
        if includeHs:  # include neighboring H atoms in functional groups
            findices_incl_h = []
            for fi in findices:
                hidx = [
                    n.GetIdx()
                    for n in mol.GetAtomWithIdx(fi).GetNeighbors()
                    if n.GetSymbol() == "H"
                ]
                findices_incl_h.extend([fi] + hidx)
            findices = findices_incl_h
        if len(findices) > 0:
            try:
                mask[torch.tensor(findices)] = 1
            except Exception as e:
                print(e)
        return mask

    return [func_groups_per_mol(mol, includeHs) for mol in to_mols]


def extract_linkers(to_mols: list[Chem.Mol]):
    def linker_per_mol(mol):
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Linker could not be extracted as molecule could not be sanitized. Skipping."
            )
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Retain original atom indices
        for a in mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())
        scaffold = GetScaffoldForMol(mol)
        if scaffold is None:
            print("Scaffold could not be extracted. Returning zero mask.")
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Collect indices for all atoms in rings
        ring_atoms = set()
        for ring in mol.GetRingInfo().AtomRings():
            ring_atoms.update(ring)
        # Define linker atoms as atoms in the scaffold that are not in any ring
        linker_atoms = [
            a.GetIntProp("org_idx")
            for a in scaffold.GetAtoms()
            if a.GetIdx() not in ring_atoms
        ]
        if not linker_atoms:
            # No linker atoms found, return a zero mask (i.e. mask with all zeros)
            # print("No linker atoms found. Returning zero mask.")
            return torch.zeros(mol.GetNumAtoms(), dtype=bool)
        # Otherwise, start with a mask that is 1 for every atom, then unmask linker atoms (set them to 0)
        mask = torch.ones(mol.GetNumAtoms(), dtype=bool)
        try:
            mask[torch.tensor(linker_atoms)] = 0
        except Exception as e:
            print(e)
        return mask

    return [linker_per_mol(mol) for mol in to_mols]


def extract_scaffolds(to_mols: list[Chem.Mol]):
    def scaffold_per_mol(mol):
        mask = torch.zeros(mol.GetNumAtoms(), dtype=bool)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            print(
                "Scaffold could not be extracted as reference molecule could not be sanitized. Skipping."
            )
            return mask
        for a in mol.GetAtoms():
            a.SetIntProp("org_idx", a.GetIdx())

        scaffold = GetScaffoldForMol(mol)
        scaffold_atoms = [a.GetIntProp("org_idx") for a in scaffold.GetAtoms()]
        if len(scaffold_atoms) > 0:
            try:
                mask[torch.tensor(scaffold_atoms)] = 1
            except Exception as e:
                print(e)
        return mask

    return [scaffold_per_mol(mol) for mol in to_mols]


def sample_mol_sizes(
    molecule_size,
    dataset,
    by_mean_and_std=False,
    upper_bound=0.1,
    lower_bound=0.1,
    n_molecules=1,
    seed=None,
    default_max=80,
    default_min=5,
):
    """
    Sample molecule sizes either by mean and std dev or by a fixed range
    :param molecule_size: The given molecule size
    :param dataset: The dataset used for sampling
    :param by_mean_and_std: If True (default: False), sample from a distribution with mean and std dev based on the dataset
    :param upper_bound: If by_mean_and_std is False, sample around the molecule size with this upper bound (percentage of molecule size)
    :param lower_bound: If by_mean_and_std is False, sample around the molecule size with this lower bound (percentage of molecule size)
    :param n_molecules: The number of molecule sizes to sample
    :param seed: The seed for reproducibility
    :return: list of sampled molecule sizes
    """

    if seed is not None:
        # set the seed for reproducibility
        np.random.seed(seed)

    max_size = (
        PLINDER_MOLECULE_SIZE_MAX
        if dataset == "plinder"
        else (
            KINODATA_MOLECULE_SIZE_MAX
            if dataset == "kinodata"
            else (
                CROSSDOCKED_MOLECULE_SIZE_MAX
                if dataset == "crossdocked"
                else default_max
            )
        )
    )
    min_size = (
        PLINDER_MOLECULE_SIZE_MIN
        if dataset == "plinder"
        else (
            KINODATA_MOLECULE_SIZE_MIN
            if dataset == "kinodata"
            else (
                CROSSDOCKED_MOLECULE_SIZE_MIN
                if dataset == "crossdocked"
                else default_min
            )
        )
    )

    if by_mean_and_std:
        mean = (
            PLINDER_MOLECULE_SIZE_MEAN
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_MEAN
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_MEAN if dataset == "crossdocked" else None
                )
            )
        )
        std_dev = (
            PLINDER_MOLECULE_SIZE_STD_DEV
            if dataset == "plinder"
            else (
                KINODATA_MOLECULE_SIZE_STD_DEV
                if dataset == "kinodata"
                else (
                    CROSSDOCKED_MOLECULE_SIZE_STD_DEV
                    if dataset == "crossdocked"
                    else None
                )
            )
        )
        if mean is None:
            raise ValueError(f"Invalid dataset {dataset}")

        if molecule_size < std_dev:
            lower_bound = molecule_size
            upper_bound = molecule_size + std_dev / 2
        elif molecule_size < mean - std_dev:
            lower_bound = molecule_size - std_dev / 2
            upper_bound = molecule_size + std_dev
        elif molecule_size > mean + std_dev:
            lower_bound = molecule_size - std_dev
            upper_bound = molecule_size + std_dev / 2
        else:
            lower_bound = molecule_size - std_dev
            upper_bound = molecule_size + std_dev

        sampled_sizes = np.random.uniform(lower_bound, upper_bound, n_molecules)
        sampled_sizes = np.round(sampled_sizes).astype(int)
        sampled_sizes = np.clip(sampled_sizes, min_size, max_size)

    else:
        sampled_sizes = np.random.uniform(
            molecule_size - lower_bound * molecule_size,
            molecule_size + upper_bound * molecule_size,
            n_molecules,
        )
        sampled_sizes = np.round(sampled_sizes).astype(int)
        sampled_sizes = np.clip(sampled_sizes, min_size, max_size)

    if n_molecules == 1:
        return sampled_sizes[0]
    else:
        return sampled_sizes


class Interpolant(ABC):
    @property
    @abstractmethod
    def hparams(self):
        pass

    @abstractmethod
    def interpolate(self, to_batch: list[SmolMol]) -> _InterpT:
        pass


class NoiseSampler(ABC):
    @property
    def hparams(self):
        pass

    @abstractmethod
    def sample_molecule(self, num_atoms: int) -> SmolMol:
        pass

    @abstractmethod
    def sample_batch(self, num_atoms: list[int]) -> SmolBatch:
        pass


class GeometricNoiseSampler(NoiseSampler):
    def __init__(
        self,
        vocab_size: int,
        n_bond_types: int,
        coord_noise: str = "gaussian",
        type_noise: str = "uniform-sample",
        bond_noise: str = "uniform-sample",
        zero_com: bool = True,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        atom_types_distribution: Optional[torch.Tensor] = None,
        bond_types_distribution: Optional[torch.Tensor] = None,
    ):
        if coord_noise != "gaussian":
            raise NotImplementedError(f"Coord noise {coord_noise} is not supported.")

        self._check_cat_noise_type(type_noise, type_mask_index, "type")
        self._check_cat_noise_type(bond_noise, bond_mask_index, "bond")

        self.vocab_size = vocab_size
        self.n_bond_types = n_bond_types
        self.coord_noise = coord_noise
        self.type_noise = type_noise
        self.bond_noise = bond_noise
        self.zero_com = zero_com
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.atom_types_distribution = atom_types_distribution
        self.bond_types_distribution = bond_types_distribution

        self.coord_dist = torch.distributions.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )

    @property
    def hparams(self):
        return {
            "coord-noise": self.coord_noise,
            "type-noise": self.type_noise,
            "bond-noise": self.bond_noise,
            "zero-com": self.zero_com,
        }

    def inpaint_molecule(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        mask: torch.Tensor,
    ) -> GeometricMol:

        inp_coords, inp_atomics = (
            from_mol.coords.clone(),
            from_mol.atomics.clone(),
        )
        inp_coords[mask, :] = to_mol.coords[mask, :]
        inp_atomics[mask, :] = to_mol.atomics[mask, :]

        # Overwrite bond types
        N = from_mol.seq_length
        bond_indices = torch.ones((N, N), device=from_mol.coords.device).nonzero(
            as_tuple=False
        )
        num_bond_types = to_mol.adjacency.size(-1)
        from_adj = torch.argmax(from_mol.adjacency, dim=-1)
        to_adj = torch.argmax(to_mol.adjacency, dim=-1)

        ## only update bonds if both atoms are inpainted and if they are bonded
        fixed_mask_matrix = (mask.unsqueeze(0) & mask.unsqueeze(1)) & (to_adj != 0)
        new_adj = from_adj.clone()
        new_adj[fixed_mask_matrix] = to_adj[fixed_mask_matrix]
        new_bond_types = smolF.one_hot_encode_tensor(
            new_adj, num_bond_types
        )  # shape: (N, N, num_bond_types)
        new_bond_types = new_bond_types[bond_indices[:, 0], bond_indices[:, 1]]

        return GeometricMol(
            inp_coords,
            inp_atomics,
            bond_indices=bond_indices,
            bond_types=new_bond_types,
        )

    def sample_molecule_inpainting(
        self,
        to_mol: GeometricMol,
        from_mol: GeometricMol,
        interaction_mask: torch.Tensor,
    ) -> GeometricMol:
        n_atoms = to_mol.seq_length
        num_fixed_atoms = interaction_mask.sum().item()

        # Reference
        coords, atomics, bond_types, bond_indices = (
            to_mol.coords,
            to_mol.atomics,
            to_mol.bond_types,
            to_mol.bond_indices,
        )
        if interaction_mask.any():
            fixed_centroid = coords[interaction_mask].mean(dim=0)
        else:
            fixed_centroid = torch.zeros(3)
        new_coords = coords.clone()
        new_coords[~interaction_mask] = fixed_centroid + self.coord_dist.sample(
            (n_atoms - num_fixed_atoms, 3)
        )

        # Sample atom types
        assert (
            self.type_noise == "uniform-sample"
        ), "Only uniform sampling of atom types is supported for inpainting"

        new_atomics = atomics.clone()
        new_atomics = new_atomics.argmax(-1)
        new_atomics[~interaction_mask] = torch.randint(
            1, self.vocab_size, (n_atoms - num_fixed_atoms,)
        )
        new_atomics = smolF.one_hot_encode_tensor(new_atomics, self.vocab_size)

        # Create bond indices and sample bond types
        bond_fixed_mask = (
            interaction_mask[bond_indices[:, 0]] & interaction_mask[bond_indices[:, 1]]
        )
        non_fixed_bond_indices = torch.nonzero(~bond_fixed_mask).squeeze(-1)

        new_bond_types = bond_types.argmax(-1).clone()
        new_bond_types[non_fixed_bond_indices] = torch.randint(
            0, self.n_bond_types, size=(non_fixed_bond_indices.shape[0],)
        )
        new_bond_types = smolF.one_hot_encode_tensor(new_bond_types, self.n_bond_types)

        # Create smol mol object
        mol = GeometricMol(
            new_coords,
            new_atomics,
            bond_indices=bond_indices,
            bond_types=new_bond_types,
        )
        return mol

    def sample_molecule(self, n_atoms: int) -> GeometricMol:

        # Sample coords and scale, if required
        coords = self.coord_dist.sample((n_atoms, 3))

        # Sample atom types
        if self.type_noise == "mask":
            atomics = torch.zeros((n_atoms, self.vocab_size), dtype=torch.float32)
            atomics[:, self.type_mask_index] = 1.0

        elif self.type_noise == "uniform-sample":
            atomics = torch.randint(2, self.vocab_size, (n_atoms,))
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)

        elif self.type_noise == "prior-sample":
            atom_types_distribution = torch.zeros(
                (self.vocab_size,), dtype=torch.float32
            )
            atom_types_distribution[1:] = (
                self.atom_types_distribution + 1.0e-6
            )  # skip pad tokens and add a bit of signal to all states
            atomics = torch.multinomial(
                atom_types_distribution, n_atoms, replacement=True
            )
            atomics = smolF.one_hot_encode_tensor(atomics, self.vocab_size)

        # Create bond indices and sample bond types
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        n_bonds = bond_indices.size(0)

        if self.bond_noise == "mask":
            bond_types = torch.tensor(self.bond_mask_index).repeat(n_bonds)
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        elif self.bond_noise == "uniform-sample":
            bond_types = torch.randint(0, self.n_bond_types, size=(n_bonds,))
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        elif self.bond_noise == "prior-sample":
            bond_types = torch.multinomial(
                self.bond_types_distribution, n_bonds, replacement=True
            )
            bond_types = smolF.one_hot_encode_tensor(bond_types, self.n_bond_types)

        # Create smol mol object
        mol = GeometricMol(
            coords,
            atomics,
            bond_indices=bond_indices,
            bond_types=bond_types,
        )
        if self.zero_com:
            mol = mol.zero_com()

        return mol

    def sample_batch(self, num_atoms: list[int]) -> GeometricMolBatch:
        mols = [self.sample_molecule(n) for n in num_atoms]
        batch = GeometricMolBatch.from_list(mols)
        return batch

    def _check_cat_noise_type(self, noise_type, mask_index, name):
        if noise_type not in ["mask", "uniform-sample", "prior-sample"]:
            raise ValueError(f"{name} noise {noise_type} is not supported.")

        if noise_type == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if {name}_noise is 'mask'."
            )


class MixedTimeSampler:
    def __init__(self, alpha, beta, mix_prob=0.5):
        self.beta_dist = torch.distributions.Beta(alpha, beta)
        self.mix_prob = mix_prob

    def sample(self, sample_shape):
        beta_sample = self.beta_dist.sample(sample_shape)
        uniform_sample = torch.rand(sample_shape)
        mix_mask = torch.rand(sample_shape) < self.mix_prob
        return torch.where(mix_mask, uniform_sample, beta_sample)


class GeometricInterpolant(Interpolant):
    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        coord_interpolation: str = "linear",
        type_interpolation: str = "unmask",
        bond_interpolation: str = "unmask",
        coord_noise_std: float = 0.0,
        type_dist_temp: float = 1.0,
        equivariant_ot: bool = False,
        batch_ot: bool = False,
        time_alpha: float = 1.0,
        time_beta: float = 1.0,
        dataset: str = "geom-drugs",
        fixed_time: Optional[float] = None,
        split_continuous_discrete_time: bool = False,
        mixed_uniform_beta_time: bool = False,
        sample_mol_sizes: Optional[bool] = False,
    ):

        if fixed_time is not None and (fixed_time < 0 or fixed_time > 1):
            raise ValueError("fixed_time must be between 0 and 1 if provided.")

        if coord_interpolation != "linear":
            raise ValueError(
                f"coord interpolation '{coord_interpolation}' not supported."
            )

        if type_interpolation not in ["unmask", "sample"]:
            raise ValueError(
                f"type interpolation '{type_interpolation}' not supported."
            )

        if bond_interpolation not in ["unmask", "sample"]:
            raise ValueError(
                f"bond interpolation '{bond_interpolation}' not supported."
            )

        self.prior_sampler = prior_sampler
        self.coord_interpolation = coord_interpolation
        self.type_interpolation = type_interpolation
        self.bond_interpolation = bond_interpolation
        self.coord_noise_std = coord_noise_std
        self.type_dist_temp = type_dist_temp
        self.equivariant_ot = equivariant_ot
        self.batch_ot = batch_ot
        self.time_alpha = time_alpha if fixed_time is None else None
        self.time_beta = time_beta if fixed_time is None else None
        self.fixed_time = fixed_time
        self.split_continuous_discrete_time = split_continuous_discrete_time
        self.sample_mol_sizes = sample_mol_sizes
        self.dataset = dataset

        if mixed_uniform_beta_time:
            self.time_dist = MixedTimeSampler(alpha=1.9, beta=1.0, mix_prob=0.02)
        else:
            self.time_dist = torch.distributions.Beta(time_alpha, time_beta)

    @property
    def hparams(self):
        prior_hparams = {f"prior-{k}": v for k, v in self.prior_sampler.hparams.items()}
        hparams = {
            "coord-interpolation": self.coord_interpolation,
            "type-interpolation": self.type_interpolation,
            "bond-interpolation": self.bond_interpolation,
            "coord-noise-std": self.coord_noise_std,
            "type-dist-temp": self.type_dist_temp,
            "equivariant-ot": self.equivariant_ot,
            "batch-ot": self.batch_ot,
            "time-alpha": self.time_alpha,
            "time-beta": self.time_beta,
            "dataset": self.dataset,
            **prior_hparams,
        }

        if self.fixed_time is not None:
            hparams["fixed-interpolation-time"] = self.fixed_time

        return hparams

    def interpolate(
        self, to_mols: list[GeometricMol], inpaint_mask: torch.Tensor = None
    ) -> _GeometricInterpT:
        batch_size = len(to_mols)
        if not self.sample_mol_sizes:
            mol_sizes = [mol.seq_length for mol in to_mols]
        else:
            mol_sizes = [
                sample_mol_sizes(mol.seq_length, self.dataset) for mol in to_mols
            ]
        num_atoms = max(mol_sizes)

        # Within match_mols either just truncate noise to match size of data molecule
        # Or also permute and rotate the noise to best match data molecule
        from_mols = [self.prior_sampler.sample_molecule(num_atoms) for mol in to_mols]
        from_mols = [
            self._match_mols(from_mol, to_mol, mol_size=mol_size)
            for from_mol, to_mol, mol_size in zip(from_mols, to_mols, mol_sizes)
        ]
        if inpaint_mask and len(inpaint_mask) > 0:
            from_mols = [
                (
                    self.prior_sampler.inpaint_molecule(to_mol, from_mol, mask)
                    if mask is not None and mask.any()
                    else from_mol
                )
                for to_mol, from_mol, mask in zip(to_mols, from_mols, inpaint_mask)
            ]

        if self.fixed_time is not None:
            times_cont = torch.tensor([self.fixed_time] * batch_size)
            times_disc = torch.tensor([self.fixed_time] * batch_size)
        else:
            times_cont = self.time_dist.sample((batch_size,))
            if self.split_continuous_discrete_time:
                times_disc = self.time_dist.sample((batch_size,))
            else:
                times_disc = times_cont.clone()

        tuples = zip(from_mols, to_mols, times_cont.tolist(), times_disc.tolist())

        interp_mols = (
            [
                self._interpolate_mol(from_mol, to_mol, t_cont=t_cont, t_disc=t_disc)
                for from_mol, to_mol, t_cont, t_disc in tuples
            ]
            if not self.sample_mol_sizes
            else from_mols
        )
        return from_mols, to_mols, interp_mols, list(times_cont), list(times_disc)

    def _ot_map(
        self, from_mols: list[GeometricMol], to_mols: list[GeometricMol]
    ) -> list[GeometricMol]:
        """Permute the from_mols batch so that it forms an approximate mini-batch OT map with to_mols"""

        mol_matrix = []
        cost_matrix = []

        # Create matrix with to mols on outer axis and from mols on inner axis
        for to_mol in to_mols:
            best_from_mols = [
                self._match_mols(from_mol, to_mol) for from_mol in from_mols
            ]
            best_costs = [self._match_cost(mol, to_mol) for mol in best_from_mols]
            mol_matrix.append(list(best_from_mols))
            cost_matrix.append(list(best_costs))

        row_indices, col_indices = linear_sum_assignment(np.array(cost_matrix))
        best_from_mols = [mol_matrix[r][c] for r, c in zip(row_indices, col_indices)]
        return best_from_mols

    def _match_mols(
        self, from_mol: GeometricMol, to_mol: GeometricMol, mol_size: int
    ) -> GeometricMol:
        """Permute the from_mol to best match the to_mol and return the permuted from_mol"""

        if mol_size > from_mol.seq_length:
            raise RuntimeError("from_mol must have at least as many atoms as to_mol.")

        # Find best permutation first, then best rotation
        # As done in Equivariant Flow Matching (https://arxiv.org/abs/2306.15030)

        # Keep the same number of atoms as the data mol in the noise mol
        mol_size = list(range(mol_size))
        from_mol = from_mol.permute(mol_size)

        if not self.equivariant_ot:
            return from_mol

        assert (
            not self.sample_mol_sizes
        ), "Cannot use equivariant OT with sampled molecule sizes"
        cost_matrix = smolF.inter_distances(
            to_mol.coords.cpu(), from_mol.coords.cpu(), sqrd=True
        )
        _, from_mol_indices = linear_sum_assignment(cost_matrix.numpy())
        from_mol = from_mol.permute(from_mol_indices.tolist())

        padded_coords = smolF.pad_tensors([from_mol.coords.cpu(), to_mol.coords.cpu()])
        from_mol_coords = padded_coords[0].numpy()
        to_mol_coords = padded_coords[1].numpy()

        rotation, _ = Rotation.align_vectors(to_mol_coords, from_mol_coords)
        from_mol = from_mol.rotate(rotation)

        return from_mol

    def _match_cost(self, from_mol: GeometricMol, to_mol: GeometricMol) -> float:
        """Calculate MSE between mol coords as a match cost"""

        sqrd_dists = smolF.inter_distances(
            from_mol.coords.cpu(), to_mol.coords.cpu(), sqrd=True
        )
        mse = sqrd_dists.mean().item()
        return mse

    def _interpolate_mol(
        self, from_mol: GeometricMol, to_mol: GeometricMol, t_cont: float, t_disc: float
    ) -> GeometricMol:
        """Interpolates mols which have already been sampled according to OT map, if required"""

        if from_mol.seq_length != to_mol.seq_length:
            raise RuntimeError(
                "Both molecules to be interpolated must have the same number of atoms."
            )

        # Interpolate coords and add gaussian noise
        coords_mean = (from_mol.coords * (1 - t_cont)) + (to_mol.coords * t_cont)
        coords_noise = torch.randn_like(coords_mean) * self.coord_noise_std
        coords = coords_mean + coords_noise

        # Interpolate atom types using unmasking or sampling
        if self.type_interpolation == "unmask":
            atom_mask = torch.rand(from_mol.seq_length) > t_disc
            to_atomics = torch.argmax(to_mol.atomics, dim=-1)
            from_atomics = torch.argmax(from_mol.atomics, dim=-1)
            to_atomics[atom_mask] = from_atomics[atom_mask]
            atomics = smolF.one_hot_encode_tensor(to_atomics, to_mol.atomics.size(-1))

        elif self.type_interpolation == "sample":
            atomics_mean = (from_mol.atomics * (1 - t_disc)) + (to_mol.atomics * t_disc)
            atomics_sample = torch.distributions.Categorical(atomics_mean).sample()
            atomics = smolF.one_hot_encode_tensor(
                atomics_sample, to_mol.atomics.size(-1)
            )

        # Interpolate bonds using unmasking or sampling
        if self.bond_interpolation == "unmask":
            to_adj = torch.argmax(to_mol.adjacency, dim=-1)
            from_adj = torch.argmax(from_mol.adjacency, dim=-1)
            bond_mask = torch.rand_like(from_adj.float()) > t_disc
            to_adj[bond_mask] = from_adj[bond_mask]
            interp_adj = smolF.one_hot_encode_tensor(to_adj, to_mol.adjacency.size(-1))

        elif self.bond_interpolation == "sample":
            adj_mean = (from_mol.adjacency * (1 - t_disc)) + (to_mol.adjacency * t_disc)
            adj_sample = torch.distributions.Categorical(adj_mean).sample()
            interp_adj = smolF.one_hot_encode_tensor(
                adj_sample, to_mol.adjacency.size(-1)
            )

        bond_indices = torch.ones((from_mol.seq_length, from_mol.seq_length)).nonzero()
        bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]

        interp_mol = GeometricMol(
            coords, atomics, bond_indices=bond_indices, bond_types=bond_types
        )
        return interp_mol


class ComplexInterpolant(GeometricInterpolant):
    """Provides apo-holo and noise to ligand interpolation by wrapping a ligand interpolant"""

    def __init__(
        self,
        prior_sampler: GeometricNoiseSampler,
        ligand_coord_interpolation="linear",
        ligand_type_interpolation="unmask",
        ligand_bond_interpolation="unmask",
        ligand_coord_noise_std: float = 0.0,
        ligand_time_alpha: float = 1.0,
        ligand_time_beta: float = 1.0,
        ligand_fixed_time: Optional[float] = None,
        split_continuous_discrete_time: bool = False,
        pocket_time_alpha: float = 1.0,
        pocket_time_beta: float = 1.0,
        pocket_fixed_time: Optional[float] = None,
        interaction_time_alpha: float = 1.0,
        interaction_time_beta: float = 1.0,
        interaction_fixed_time: Optional[float] = None,
        pocket_coord_noise_std: float = 0.0,
        rigid_pocket: bool = False,
        separate_pocket_interpolation: bool = False,
        separate_interaction_interpolation: bool = False,
        n_interaction_types: Optional[int] = None,
        flow_interactions: bool = False,
        dataset: str = "plinder",
        sample_mol_sizes: Optional[bool] = False,
        interaction_inpainting: bool = False,
        scaffold_inpainting: bool = False,
        func_group_inpainting: bool = False,
        linker_inpainting: bool = False,
        max_fragment_cuts: int = 3,
        fragment_inpainting: bool = False,
        substructure_inpainting: bool = False,
        substructure: Optional[str] = None,
        mixed_uncond_inpaint: bool = False,
        mixed_uniform_beta_time: bool = False,
        equivariant_ot: bool = False,
        batch_ot: bool = False,
        inference: bool = False,
        vocab: Optional[dict[str, int]] = None,
    ):

        super().__init__(
            prior_sampler,
            coord_interpolation=ligand_coord_interpolation,
            type_interpolation=ligand_type_interpolation,
            bond_interpolation=ligand_bond_interpolation,
            coord_noise_std=ligand_coord_noise_std,
            equivariant_ot=equivariant_ot,
            batch_ot=batch_ot,
            time_alpha=ligand_time_alpha,
            time_beta=ligand_time_beta,
            fixed_time=ligand_fixed_time,
            split_continuous_discrete_time=split_continuous_discrete_time,
            mixed_uniform_beta_time=mixed_uniform_beta_time,
            dataset=dataset,
            sample_mol_sizes=sample_mol_sizes,
        )
        if sample_mol_sizes:
            print("Running inference with sampled molecule sizes!")

        self.rigid_pocket = rigid_pocket
        self.separate_pocket_interpolation = separate_pocket_interpolation
        self.separate_interaction_interpolation = separate_interaction_interpolation
        self.pocket_coord_noise_std = pocket_coord_noise_std
        self.pocket_time_alpha = pocket_time_alpha
        self.pocket_time_beta = pocket_time_beta
        self.pocket_fixed_time = pocket_fixed_time
        self.interaction_time_alpha = interaction_time_alpha
        self.interaction_time_beta = interaction_time_beta
        self.interaction_fixed_time = interaction_fixed_time
        self.flow_interactions = flow_interactions
        self.n_interaction_types = n_interaction_types
        self.interaction_inpainting = interaction_inpainting
        self.scaffold_inpainting = scaffold_inpainting
        self.func_group_inpainting = func_group_inpainting
        self.linker_inpainting = linker_inpainting
        self.fragment_inpainting = fragment_inpainting
        self.max_fragment_cuts = max_fragment_cuts
        self.substructure_inpainting = substructure_inpainting
        self.substructure = substructure
        self.mixed_uncond_inpaint = mixed_uncond_inpaint
        self.sample_mol_sizes = sample_mol_sizes

        self.inference = inference
        self.vocab = vocab

        self.pocket_time_dist = torch.distributions.Beta(
            pocket_time_alpha, pocket_time_beta
        )
        self.interaction_time_dist = torch.distributions.Beta(
            interaction_time_alpha, interaction_time_beta
        )

    @property
    def hparams(self):
        ligand_hparams = {f"ligand-{k}": v for k, v in super().hparams.items()}
        hparams = {
            "rigid-pocket": self.rigid_pocket,
            "separate-pocket-interpolation": self.separate_pocket_interpolation,
            "pocket-coord-noise-std": self.pocket_coord_noise_std,
            **ligand_hparams,
        }
        hparams["separate-interaction-interpolation"] = (
            self.separate_interaction_interpolation
        )
        hparams["n-interaction-types"] = self.n_interaction_types
        hparams["flow-interactions"] = self.flow_interactions
        hparams["interaction-inpainting"] = self.interaction_inpainting
        hparams["scaffold-inpainting"] = self.scaffold_inpainting
        hparams["func-group-inpainting"] = self.func_group_inpainting
        hparams["mixed-uncond-inpaint"] = self.mixed_uncond_inpaint

        if self.separate_pocket_interpolation:
            hparams["pocket-time-alpha"] = self.pocket_time_alpha
            hparams["pocket-time-beta"] = self.pocket_time_beta
            if self.pocket_fixed_time is not None:
                hparams["pocket-fixed-interpolation-time"] = self.fixed_time

        if self.separate_interaction_interpolation:
            hparams["interaction-time-alpha"] = self.interaction_time_alpha
            hparams["interaction-time-beta"] = self.interaction_time_beta
            if self.interaction_fixed_time is not None:
                hparams["interaction-fixed-interpolation-time"] = self.fixed_time

        return hparams

    # NOTE the apo and holo pairs must come with 1-1 match on atoms and bonds, except coordinate values
    # NOTE this also assumes that each system has already been shifted so that the apo pocket has a zero com
    def interpolate(self, to_mols: list[PocketComplex]) -> _ComplexInterpT:
        batch_size = len(to_mols)

        # Interpolate ligands
        inpaint_mask = []
        if (
            self.interaction_inpainting
            or self.scaffold_inpainting
            or self.func_group_inpainting
            or self.linker_inpainting
            or self.substructure_inpainting
            or self.fragment_inpainting
        ):
            assert (
                not self.sample_mol_sizes
            ), "Inpainting currently not supported with sampled mol sizes"
            if self.interaction_inpainting:
                interaction_mask = [
                    mol.interactions[:, :, 1:].sum(dim=(0, 2)) > 0 for mol in to_mols
                ]
            if self.scaffold_inpainting:
                scaffold_mask = extract_scaffolds(
                    [mol.ligand.to_rdkit(vocab=self.vocab) for mol in to_mols]
                )
            if self.func_group_inpainting:
                func_group_mask = extract_func_groups(
                    [mol.ligand.to_rdkit(vocab=self.vocab) for mol in to_mols]
                )
            if self.substructure_inpainting:
                assert (
                    self.substructure is not None
                ), "Substructure query must be provided"
                assert isinstance(self.substructure, str) or isinstance(
                    self.substructure, list
                ), "Substructure must be a string or a list of indices"
                custom_mask = extract_substructure(
                    [mol.ligand.to_rdkit(vocab=self.vocab) for mol in to_mols],
                    substructure_query=self.substructure,
                )
            if self.linker_inpainting:
                linker_mask = extract_linkers(
                    [mol.ligand.to_rdkit(vocab=self.vocab) for mol in to_mols]
                )
            if self.fragment_inpainting:
                fragment_mask = extract_fragments(
                    [mol.ligand.to_rdkit(vocab=self.vocab) for mol in to_mols],
                    maxCuts=self.max_fragment_cuts,
                )

            # Randomly select one of the mask for both training and inference (if not specified otherwise)
            masks = []
            if self.interaction_inpainting:
                masks.append(interaction_mask)
            if self.scaffold_inpainting:
                masks.append(scaffold_mask)
            if self.func_group_inpainting:
                masks.append(func_group_mask)
            if self.substructure_inpainting:
                masks.append(custom_mask)
            if self.linker_inpainting:
                masks.append(linker_mask)
            if self.fragment_inpainting:
                masks.append(fragment_mask)

            inpaint_mask = random.choice(masks)

            # if training, select inpainting mask with 50% or set to False
            if not self.inference and self.mixed_uncond_inpaint:
                inpaint_mask = [
                    m if torch.rand(1) < 0.5 else torch.tensor([0] * len(m)).bool()
                    for m in inpaint_mask
                ]

        ligands = [system.ligand for system in to_mols]
        (
            from_ligands,
            to_ligands,
            interp_ligands,
            ligand_times_cont,
            ligand_times_disc,
        ) = super().interpolate(ligands, inpaint_mask=inpaint_mask)

        # Save metadata and center-of-mass for each system
        metadata = [system.metadata for system in to_mols]
        com = [system.com for system in to_mols]

        # Interpolate interactions
        to_interactions = (
            [system.interactions for system in to_mols]
            if self.n_interaction_types is not None
            else []
        )
        if self.flow_interactions:
            assert (
                self.n_interaction_types is not None
            ), "Flowing interactions requires n_interaction_types to be specified"
            if self.separate_interaction_interpolation:
                if self.interaction_fixed_time is not None:
                    interaction_times = [
                        torch.tensor(self.interaction_fixed_time)
                    ] * batch_size
                else:
                    interaction_times = self.interaction_time_dist.sample((batch_size,))
                    interaction_times = interaction_times.tolist()
            else:
                interaction_times = ligand_times_disc
            from_interactions = [
                self._noise_interactions(
                    n_pocket_atoms=to_mols[i].holo.seq_length,
                    n_ligand_atoms=from_ligands[i].seq_length,
                    n_interactions=self.n_interaction_types,
                )
                for i in range(len(to_mols))
            ]
            if self.inference:
                interp_interactions = from_interactions
            else:
                interp_interactions = [
                    self._interpolate_interactions(from_interaction, to_interaction, t)
                    for from_interaction, to_interaction, t in zip(
                        from_interactions, to_interactions, interaction_times
                    )
                ]
        else:
            interaction_times = [torch.tensor(0)] * batch_size
            from_interactions = interp_interactions = to_interactions

        # Interpolate pockets
        if self.separate_pocket_interpolation:
            if self.pocket_fixed_time is not None:
                pocket_times = torch.tensor([self.pocket_fixed_time] * batch_size)
            else:
                pocket_times = self.pocket_time_dist.sample((batch_size,))

            pocket_times = pocket_times.tolist()
        else:
            pocket_times = ligand_times_cont

        holo_pockets = [system.holo for system in to_mols]
        apo_pockets = (
            holo_pockets if self.rigid_pocket else [system.apo for system in to_mols]
        )
        interp_pockets = [
            self._interpolate_pocket(apo_pocket, holo_pocket, t)
            for apo_pocket, holo_pocket, t in zip(
                apo_pockets, holo_pockets, pocket_times
            )
        ]

        # Combine everything back into PocketComplex objects
        from_systems = [
            PocketComplex(
                apo=apo_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
                com=_com,
            )
            for apo_pocket, ligand, interaction, meta, _mask, _com in zip_longest(
                apo_pockets,
                from_ligands,
                from_interactions,
                metadata,
                inpaint_mask,
                com,
                fillvalue=None,
            )
        ]
        to_systems = [
            PocketComplex(
                holo=holo_pocket,
                apo=apo_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
                com=_com,
            )
            for holo_pocket, apo_pocket, ligand, interaction, meta, _mask, _com in zip_longest(
                holo_pockets,
                apo_pockets,
                to_ligands,
                to_interactions,
                metadata,
                inpaint_mask,
                com,
                fillvalue=None,
            )
        ]
        interp_systems = [
            PocketComplex(
                apo=interp_pocket,
                ligand=ligand,
                interactions=interaction,
                metadata=meta,
                fragment_mask=_mask,
            )
            for interp_pocket, ligand, meta, interaction, _mask in zip_longest(
                interp_pockets,
                interp_ligands,
                metadata,
                interp_interactions,
                inpaint_mask,
                fillvalue=None,
            )
        ]

        # Save times for ligand, pocket and interaction interpolation
        times = []
        for t in zip(
            ligand_times_cont, ligand_times_disc, pocket_times, interaction_times
        ):
            times.append(torch.stack(t))

        return from_systems, to_systems, interp_systems, times

    def _interpolate_pocket(
        self, apo_pocket: ProteinPocket, holo_pocket: ProteinPocket, t: torch.Tensor
    ) -> ProteinPocket:
        assert len(apo_pocket) == len(
            holo_pocket
        ), "apo and holo pockets must have the same number of atoms"

        # Interpolate coords and add gaussian noise
        # Apo and holo should come pre-aligned so no need for any alignment here
        coords_mean = (apo_pocket.mol.coords * (1 - t)) + (holo_pocket.mol.coords * t)
        coords_noise = torch.randn_like(coords_mean) * self.pocket_coord_noise_std
        coords = coords_mean + coords_noise

        # NOTE Assumes apo and holo have a 1-1 match on everything except coordinates
        interp_pocket_mol = holo_pocket.mol._copy_with(coords=coords)
        interp_pocket = holo_pocket._copy_with(mol=interp_pocket_mol)

        return interp_pocket

    def _interpolate_interactions(
        self,
        from_interactions: torch.Tensor,
        to_interactions: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Interpolate interactions

        num_interactions = to_interactions.size(-1)
        to_inter = torch.argmax(to_interactions, dim=-1)
        from_inter = torch.argmax(from_interactions, dim=-1)
        interaction_mask = torch.rand_like(from_inter.float()) > t
        to_inter[interaction_mask] = from_inter[interaction_mask]
        interp_interactions = smolF.one_hot_encode_tensor(to_inter, num_interactions)
        return interp_interactions

    def _noise_interactions(
        self, n_pocket_atoms: int, n_ligand_atoms: int, n_interactions: int
    ):
        num_pairs = n_pocket_atoms * n_ligand_atoms
        prior_flat = torch.zeros((num_pairs, n_interactions))
        prior_interactions = torch.randint(0, n_interactions, size=(num_pairs,))
        prior_flat[torch.arange(num_pairs), prior_interactions] = 1.0
        from_interaction = prior_flat.reshape(
            n_pocket_atoms, n_ligand_atoms, n_interactions
        )

        return from_interaction

    def _interaction_ot_map(self, from_systems, from_ligands, from_interactions):
        """
        Permute the from_ligands batch so that it forms an approximate mini-batch OT map with from_interactions:
        Meaning, align the from_ligands batch with the from_systems batch, such that the ligand atoms that are
        involved in the interactions between pocket and ligand are close to the pocket atoms that are involved
        in the interactions between pocket and ligand given by the from_interactions batch.
        Hence, permute ligand atoms based on the distance to the pocket atoms that show interactions with the ligand atoms.
        """
        raise NotImplementedError("Interaction OT map not implemented yet.")

        def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
            """
            Compute the optimal rotation matrix that aligns Q onto P
            using the Kabsch algorithm (no reflection allowed).
            """
            P_mean = P.mean(axis=0)
            Q_mean = Q.mean(axis=0)
            P_centered = P - P_mean
            Q_centered = Q - Q_mean

            # Covariance matrix
            C = np.dot(Q_centered.T, P_centered)  # (3xK) dot (Kx3) -> (3x3)

            # SVD
            V, S, Wt = np.linalg.svd(C)
            d = np.linalg.det(np.dot(V, Wt))
            if d < 0.0:
                V[:, -1] = -V[:, -1]
            R = np.dot(V, Wt)
            return R

        def align_points_kabsch(
            P: np.ndarray, Q: np.ndarray, all_Q: np.ndarray
        ) -> np.ndarray:
            """
            Given matched points P, Q of shape (K, 3), compute the rigid transform
            (rotation + translation) that best aligns Q to P. Then apply that transform
            to all_Q (shape (N, 3)), returning the aligned points of same shape (N, 3).
            """
            P_mean = P.mean(axis=0)
            Q_mean = Q.mean(axis=0)
            R = kabsch_rotation(P, Q)

            aligned_all_Q = (all_Q - Q_mean) @ R + P_mean
            return aligned_all_Q

        def optimal_ligand_alignment(
            pocket_coords: np.ndarray,
            ligand_coords: np.ndarray,
            interaction_matrix: np.ndarray,
            large_penalty: float = 1e6,
        ) -> np.ndarray:
            """
            Find a partial matching between pocket atoms (N_p x 3) and ligand atoms (N_l x 3)
            by minimizing distance while respecting interactions, then compute
            the rigid transform that best aligns the entire ligand to the pocket.
            """
            # pairwise distance matrix (N_p x N_l)
            diffs = pocket_coords[:, None, :] - ligand_coords[None, :, :]
            distance_matrix = np.linalg.norm(diffs, axis=2)

            # interaction mask
            feasible_mask = np.any(interaction_matrix, axis=2)  # (N_p, N_l)
            cost_matrix = np.where(feasible_mask, distance_matrix, large_penalty)

            # linear assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            pocket_matched = pocket_coords[row_ind]  # shape (K, 3)
            ligand_matched = ligand_coords[col_ind]  # shape (K, 3)

            # transform
            aligned_ligand_coords = align_points_kabsch(
                P=pocket_matched, Q=ligand_matched, all_Q=ligand_coords
            )
            return aligned_ligand_coords

        # aligned_ligand = optimal_ligand_alignment(
        #     pocket_coords=pocket_coords,
        #     ligand_coords=ligand_coords,
        #     interaction_matrix=to_interaction,
        #     large_penalty=1e6
        # )
