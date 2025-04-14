import os
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, OneCycleLR
from torchmetrics import MetricCollection

import flowr.util.functional as smolF
import flowr.util.metrics as Metrics
import flowr.util.rdkit as smolRD
from flowr.data.data_info import GeneralInfos as DataInfos
from flowr.models.semla import MolecularGenerator
from flowr.util.molecule import Molecule
from flowr.util.molrepr import GeometricMol
from flowr.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="mask",
        bond_strategy="mask",
        pocket_noise=None,
        ligand_only=False,
        cat_noise_level=0,
        type_mask_index=None,
        bond_mask_index=None,
        dataset="geom-drugs",
        eps=1e-5,
    ):

        self._check_cat_sampling_strategy(type_strategy, type_mask_index, "type")
        self._check_cat_sampling_strategy(bond_strategy, bond_mask_index, "bond")

        self.steps = steps
        self.coord_noise_std = coord_noise_std
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.pocket_noise = pocket_noise
        self.ligand_only = ligand_only
        self.cat_noise_level = cat_noise_level
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.dataset = dataset
        self.eps = eps

    @property
    def hparams(self):
        return {
            "integration-steps": self.steps,
            "integration-coord-noise-std": self.coord_noise_std,
            "integration-type-strategy": self.type_strategy,
            "integration-bond-strategy": self.bond_strategy,
            "integration-cat-noise-level": self.cat_noise_level,
        }

    def step(
        self, curr: _BatchT, predicted: _BatchT, prior: _BatchT, t: _T, step_size: float
    ) -> _BatchT:

        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)

        # *** Coord update step ***
        coord_velocity = (predicted["coords"] - curr["coords"]) / (1 - t.view(-1, 1, 1))
        coord_velocity += torch.randn_like(coord_velocity) * self.coord_noise_std
        coords = curr["coords"] + (step_size * coord_velocity)
        if self.pocket_noise == "random" or self.pocket_noise is None:
            coords = smolF.zero_com(coords, node_mask=prior["mask"])
        coords = coords * prior["mask"].unsqueeze(-1)

        # *** Atom type update step ***
        if self.type_strategy == "linear":
            one_hots = torch.eye(vocab_size, device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Dirichlet refers to sampling from a dirichlet dist, not dirichlet FM
        elif self.type_strategy == "dirichlet":
            type_velocity = torch.distributions.Dirichlet(
                predicted["atomics"] + self.eps
            ).sample()
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                t,
                self.type_mask_index,
                step_size,
            )

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(
                curr["atomics"], predicted["atomics"], t, step_size
            )

        # *** Bond update step ***
        if self.type_strategy == "linear":
            one_hots = torch.eye(n_bonds, device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.type_strategy == "dirichlet":
            bond_velocity = torch.distributions.Dirichlet(
                predicted["bonds"] + self.eps
            ).sample()
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(
                curr["bonds"],
                predicted["bonds"],
                t,
                self.bond_mask_index,
                step_size,
            )

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(
                curr["bonds"], predicted["bonds"], t, step_size
            )

        if (
            self.pocket_noise in ["fix", "apo"] and not self.ligand_only
        ):  # overwrite the pocket atom and bond type predictions with the holo data
            coords, atomics, bonds = self._overwrite_pocket_prediction(
                coords, atomics, bonds, prior, pocket_noise=self.pocket_noise
            )

        updated = {
            "coords": coords,
            "atomics": atomics.float(),
            "bonds": bonds.float(),
            "mask": curr["mask"],
        }
        if "lig_mask" in curr:
            updated["lig_mask"] = curr["lig_mask"]
            updated["pocket_mask"] = curr["pocket_mask"]
            updated["atom_names"] = curr["atom_names"]
            updated["res_names"] = curr["res_names"]
        return updated

    # TODO test with mask sampling
    def _mask_sampling_step(self, curr_dist, pred_dist, t, mask_index, step_size):
        n_categories = pred_dist.size(-1)

        pred = torch.distributions.Categorical(pred_dist).sample()
        curr = torch.argmax(curr_dist, dim=-1)

        ones = [1] * (len(pred.shape) - 1)
        times = t.view(-1, *ones)

        # Choose elements to unmask
        limit = step_size * (1 + (self.cat_noise_level * times)) / (1 - times)
        unmask = torch.rand_like(pred.float()) < limit
        unmask = unmask * (curr == mask_index)

        # Choose elements to mask
        mask = torch.rand_like(pred.float()) < step_size * self.cat_noise_level
        mask = mask * (curr != self.type_mask_index)
        mask[t + step_size >= 1.0] = 0.0

        # Applying unmasking and re-masking
        curr[unmask] = pred[unmask]
        curr[mask] = mask_index

        return smolF.one_hot_encode_tensor(curr, n_categories)

    def _uniform_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        curr = torch.argmax(curr_dist, dim=-1).unsqueeze(-1)
        pred_probs_curr = torch.gather(pred_dist, -1, curr)

        # Setup batched time tensor and noise tensor
        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)
        noise = torch.zeros_like(times)
        noise[times + step_size < 1.0] = self.cat_noise_level

        # Off-diagonal step probs
        mult = (1 + ((2 * noise) * (n_categories - 1) * times)) / (1 - times)
        first_term = step_size * mult * pred_dist
        second_term = step_size * noise * pred_probs_curr
        step_probs = (first_term + second_term).clamp(max=1.0)

        # On-diagonal step probs
        step_probs.scatter_(-1, curr, 0.0)
        diags = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        step_probs.scatter_(-1, curr, diags)

        # Sample and convert back to one-hot so that all strategies represent data the same way
        samples = torch.distributions.Categorical(step_probs).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "dirichlet", "mask", "uniform-sample"]:
            raise ValueError(f"{name} sampling strategy '{strategy}' is not supported.")

        if strategy == "mask" and mask_index is None:
            raise ValueError(
                f"{name}_mask_index must be provided if using the mask sampling strategy."
            )

    def _overwrite_pocket_prediction(
        self, coords, atomics, bonds, prior, pocket_noise="apo"
    ):
        pocket_mask = prior["pocket_mask"].bool()
        mol_coords_list = []
        type_probs_list = []
        bond_probs_list = []

        for idx in range(coords.size(0)):
            if self.pocket_noise == "fix":
                mol_coords = coords[idx]
                mol_coords[pocket_mask[idx]] = prior["coords"][idx][pocket_mask[idx]]
                mol_coords_list.append(mol_coords)
            mol_type_probs = atomics[idx]
            mol_type_probs[pocket_mask[idx]] = prior["atomics"][idx][
                pocket_mask[idx]
            ].long()
            type_probs_list.append(mol_type_probs)

            present_indices = pocket_mask[idx].nonzero(as_tuple=True)[0]
            mol_bond_probs = bonds[idx]
            mol_bond_probs[present_indices[:, None], present_indices] = prior["bonds"][
                idx
            ][present_indices[:, None], present_indices].long()
            bond_probs_list.append(mol_bond_probs)

        if self.pocket_noise == "fix":
            coords = torch.stack(mol_coords_list)
        atomics = torch.stack(type_probs_list)
        bonds = torch.stack(bond_probs_list)

        return coords, atomics, bonds


class MolBuilder:
    def __init__(self, vocab, pocket_noise=None, save_dir=None, n_workers=16):
        self.vocab = vocab
        self.pocket_noise = pocket_noise
        self.save_dir = save_dir
        self.n_workers = n_workers
        self._executor = None

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _startup(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(self.n_workers)

    def mols_from_smiles(self, smiles, explicit_hs=False):
        self._startup()
        futures = [
            self._executor.submit(smolRD.mol_from_smiles, smi, explicit_hs)
            for smi in smiles
        ]
        mols = [future.result() for future in futures]
        self.shutdown()
        return mols

    def mols_from_tensors(
        self,
        coords,
        atom_dists,
        mask,
        bond_dists=None,
        charge_dists=None,
        sanitise=True,
    ):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    def ligs_from_complex(
        self,
        coords,
        mask,
        atom_dists,
        bond_dists=None,
        charge_dists=None,
        sanitise=True,
    ):
        extracted = self._extract_pocket_or_lig(
            coords, mask, atom_dists, bond_dists=bond_dists, charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._mol_from_tensors, sanitise=sanitise)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        mols = [future.result() for future in futures]
        self.shutdown()

        return mols

    # def pdb_from_tensors(coords, pdb_file):

    def pockets_from_complex(
        self,
        coords,
        pocket_mask,
        atom_dists,
        bond_dists,
        charge_dists=None,
    ):

        extracted = self._extract_pocket_or_lig(
            coords,
            pocket_mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
        )

        self._startup()
        build_fn = partial(self._pocket_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        pdbs = [future.result() for future in futures]
        self.shutdown()

        return pdbs

    def _pocket_from_tensors(
        self, coords, atom_dists, bond_dists=None, charge_dists=None, pdb_path=None
    ):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = (
            self._mol_extract_charges(charge_dists)
            if charge_dists is not None
            else None
        )
        mol = smolRD.mol_from_atoms(
            coords.numpy(), tokens, bonds=bonds, charges=charges, sanitise=False
        )
        if pdb_path is not None:
            from rdkit import Chem

            Chem.MolToPDBFile(mol, pdb_path)

        return mol

    # TODO move into from_tensors method of GeometricMolBatch
    def smol_from_tensors(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )

        self._startup()
        build_fn = partial(self._smol_from_tensors)
        futures = [self._executor.submit(build_fn, *items) for items in extracted]
        smol_mols = [future.result() for future in futures]
        self.shutdown()

        return smol_mols

    def _mol_from_tensors(
        self, coords, atom_dists, bond_dists=None, charge_dists=None, sanitise=True
    ):
        tokens = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists) if bond_dists is not None else None
        charges = (
            self._mol_extract_charges(charge_dists)
            if charge_dists is not None
            else None
        )
        return smolRD.mol_from_atoms(
            coords.numpy(), tokens, bonds=bonds, charges=charges, sanitise=sanitise
        )

    def _smol_from_tensors(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.size(0)

        charges = torch.tensor(self._mol_extract_charges(charge_dists))
        bond_indices = torch.ones((n_atoms, n_atoms)).nonzero()
        bond_types = bond_dists[bond_indices[:, 0], bond_indices[:, 1], :]

        mol = GeometricMol(coords, atom_dists, bond_indices, bond_types, charges)
        return mol

    def mol_stabilities(self, coords, atom_dists, mask, bond_dists, charge_dists):
        extracted = self._extract_mols(
            coords, atom_dists, mask, bond_dists=bond_dists, charge_dists=charge_dists
        )
        mol_atom_stabilities = [self.atom_stabilities(*items) for items in extracted]
        return mol_atom_stabilities

    def atom_stabilities(self, coords, atom_dists, bond_dists, charge_dists):
        n_atoms = coords.shape[0]

        atomics = self._mol_extract_atomics(atom_dists)
        bonds = self._mol_extract_bonds(bond_dists)
        charges = self._mol_extract_charges(charge_dists).tolist()

        # Recreate the adj to ensure it is symmetric
        bond_indices = torch.tensor(bonds[:, :2])
        bond_types = torch.tensor(bonds[:, 2])
        adj = smolF.adj_from_edges(bond_indices, bond_types, n_atoms, symmetric=True)

        adj[adj == 4] = 1.5
        valencies = adj.sum(dim=-1).long()

        stabilities = []
        for i in range(n_atoms):
            atom_type = atomics[i]
            charge = charges[i]
            valence = valencies[i].item()

            if atom_type not in Metrics.ALLOWED_VALENCIES:
                stabilities.append(False)
                continue

            allowed = Metrics.ALLOWED_VALENCIES[atom_type]
            atom_stable = Metrics._is_valid_valence(valence, allowed, charge)
            stabilities.append(atom_stable)

        return stabilities

    # Separate each molecule from the batch
    def _extract_mols(
        self, coords, atom_dists, mask, bond_dists=None, charge_dists=None
    ):
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        n_atoms = mask.sum(dim=1)
        for idx in range(coords.size(0)):
            mol_atoms = n_atoms[idx]
            mol_coords = coords[idx, :mol_atoms, :].cpu()
            mol_token_dists = atom_dists[idx, :mol_atoms, :].cpu()

            coords_list.append(mol_coords)
            atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                mol_bond_dists = bond_dists[idx, :mol_atoms, :mol_atoms, :].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx, :mol_atoms, :].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    def _extract_pocket_or_lig(
        self,
        coords,
        mask,
        atom_dists=None,
        bond_dists=None,
        charge_dists=None,
    ):
        """
        Extract the ligand or pocket from the complex data
        Specify the mask to extract the ligand or pocket
        """
        coords_list = []
        atom_dists_list = []
        bond_dists_list = []
        charge_dists_list = []

        for idx in range(coords.size(0)):
            mol_coords = coords[idx][mask[idx]].cpu()
            coords_list.append(mol_coords)
            if atom_dists is not None:
                mol_token_dists = atom_dists[idx][mask[idx]].cpu()
                atom_dists_list.append(mol_token_dists)

            if bond_dists is not None:
                present_indices = mask[idx].nonzero(as_tuple=True)[0]
                mol_bond_dists = bond_dists[idx][
                    present_indices[:, None], present_indices
                ].cpu()
                bond_dists_list.append(mol_bond_dists)
            else:
                bond_dists_list.append(None)

            if charge_dists is not None:
                mol_charge_dists = charge_dists[idx][mask[idx]].cpu()
                charge_dists_list.append(mol_charge_dists)
            else:
                charge_dists_list.append(None)

        zipped = zip(coords_list, atom_dists_list, bond_dists_list, charge_dists_list)
        return zipped

    # Take index with highest probability and convert to token
    def _mol_extract_atomics(self, atom_dists):
        vocab_indices = torch.argmax(atom_dists, dim=1).tolist()
        tokens = self.vocab.tokens_from_indices(vocab_indices)
        return tokens

    # Convert to atomic number bond list format
    def _mol_extract_bonds(self, bond_dists):
        bond_types = torch.argmax(bond_dists, dim=-1)
        bonds = smolF.bonds_from_adj(bond_types)
        return bonds.long().numpy()

    # Convert index from model to actual atom charge
    def _mol_extract_charges(self, charge_dists):
        charge_types = torch.argmax(charge_dists, dim=-1).tolist()
        charges = [smolRD.IDX_CHARGE_MAP[idx] for idx in charge_types]
        return np.array(charges)

    def write_xyz_file(self, coords, atom_types, filename):
        out = f"{len(coords)}\n\n"
        assert len(coords) == len(atom_types)
        for i in range(len(coords)):
            out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
        with open(filename, "w") as f:
            f.write(out)

    def tensors_to_xyz(
        self,
        prior=None,
        interpolated=None,
        data=None,
        predicted=None,
        coord_scale=1.0,
        idx=0,
        save_dir=".",
    ):
        """
        Write the coordinates and atom types of the ligand and pocket atoms to an xyz file.
        Can be used to debug the model while training or inference to see how the ligand and pocket atoms are placed.

        idx: index of the molecule in the batch to write to file
        """
        if data is not None:
            mask = data["mask"]
            coords = data["coords"]
            atom_dists = data["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "ref.xyz"),
            )

        if interpolated is not None:
            mask = interpolated["mask"]
            coords = interpolated["coords"]
            atom_dists = interpolated["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "interpolated.xyz"),
            )

        if prior is not None:
            mask = prior["mask"]
            coords = prior["coords"]
            atom_dists = prior["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "prior.xyz"),
            )

        if predicted is not None:
            assert data is not None
            mask = data["mask"]
            coords = predicted["coords"]
            atom_dists = predicted["atomics"]
            extracted = list(
                self._extract_mols(
                    coords,
                    atom_dists,
                    mask,
                )
            )
            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, "predicted.xyz"),
            )

    def write_xyz_file_from_batch(
        self,
        data,
        coord_scale=1.0,
        path=".",
        t=0,
    ):
        if not os.path.exists(path):
            os.makedirs(path)

        mask = data["mask"]
        coords = data["coords"]
        atom_dists = data["atomics"]
        extracted = list(
            self._extract_mols(
                coords,
                atom_dists,
                mask,
            )
        )
        bs = len(coords)
        for idx in range(bs):
            save_dir = os.path.join(path, f"graph_{idx}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            coords = extracted[idx][0] * coord_scale
            atom_types = self._mol_extract_atomics(extracted[idx][1])
            self.write_xyz_file(
                coords=coords,
                atom_types=atom_types,
                filename=os.path.join(save_dir, f"latent_{t}.xyz"),
            )

    def write_trajectory_as_xyz(
        self,
        num_mols,
        file_path,
        save_path,
        remove_intermediate_files=True,
    ):

        def get_key(fp):
            filename = os.path.splitext(os.path.basename(fp))[0]
            int_part = filename.split("_")[-1]
            return int(int_part)

        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=False)
        for i in range(num_mols):
            files = sorted(
                glob(os.path.join(file_path, f"graph_{i}/latent_*.xyz")), key=get_key
            )
            traj_path = save_path / f"trajectory_{i}.xyz"
            if traj_path.is_file():
                traj_path.unlink()

            for j, file in enumerate(files):
                with open(file, "r") as f:
                    lines = f.readlines()

                with open(traj_path, "a") as file:
                    for line in lines:
                        file.write(line)
                    if (
                        j == len(files) - 1
                    ):  ####write the last timestep 10x for better visibility
                        for _ in range(10):
                            for line in lines:
                                file.write(line)

        if remove_intermediate_files:
            shutil.rmtree(file_path)

    def add_ligand_to_pocket(
        self,
        lig_data,
        lig_mask,
        complex_data,
        add_charges=False,
        add_pocket_info=True,
    ):
        coords = complex_data["coords"].clone()
        atomics = complex_data["atomics"].clone()
        bonds = complex_data["bonds"].clone()
        charges = complex_data["charges"].clone()
        mask = complex_data["lig_mask"].bool()

        lig_coords, lig_atomics, lig_bonds = (
            lig_data["coords"],
            lig_data["atomics"],
            lig_data["bonds"],
        )
        if add_charges:
            lig_charges = lig_data["charges"]
        else:
            lig_charges = None
        for idx in range(lig_coords.size(0)):
            coords[idx, mask[idx], :] = lig_coords[idx, lig_mask[idx], :]
            atomics[idx, mask[idx], :] = lig_atomics[idx, lig_mask[idx], :]
            bond_indices = mask[idx].nonzero(as_tuple=True)[0]
            bond_indices_lig = lig_mask[idx].nonzero(as_tuple=True)[0]
            bonds[idx][bond_indices[:, None], bond_indices] = lig_bonds[idx][
                bond_indices_lig[:, None], bond_indices_lig
            ]
            if add_charges:
                charges[idx, mask[idx], :] = lig_charges[idx, lig_mask[idx], :]

        out = {
            "coords": coords,
            "atomics": atomics,
            "bonds": bonds,
        }
        if add_charges:
            out["charges"] = charges
        if add_pocket_info:
            out["mask"] = complex_data["mask"]
            out["lig_mask"] = complex_data["lig_mask"]
            out["pocket_mask"] = complex_data["pocket_mask"]
            out["atom_names"] = complex_data["atom_names"]
            out["res_names"] = complex_data["res_names"]

        return out

    def extract_ligand_from_complex(self, data):
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        if "charges" in data:
            charges = data["charges"]
        lig_mask = data["lig_mask"].bool()
        max_atoms = lig_mask.sum(dim=1).max().item()

        lig_coords = []
        lig_atomics = []
        lig_charges = []
        lig_bonds = []
        for i in range(coords.size(0)):
            lig_coords.append(coords[i, lig_mask[i], :])
            lig_atomics.append(atomics[i, lig_mask[i], :])
            if "charges" in data:
                lig_charges.append(charges[i, lig_mask[i], :])
            num_atoms = lig_mask[i].sum().item()
            bond_probs = torch.zeros(max_atoms, max_atoms, bonds.shape[-1]).to(
                lig_mask.device
            )
            bond_indices = lig_mask[i].nonzero(as_tuple=True)[0]
            bond_probs[:num_atoms, :num_atoms, :] = bonds[i][
                bond_indices[:, None], bond_indices
            ]
            lig_bonds.append(bond_probs)
        atom_mask = (
            smolF.pad_tensors([torch.ones(len(coords)) for coords in lig_coords])
            .to(lig_mask.device)
            .int()
        )
        lig_coords = smolF.pad_tensors(lig_coords)
        lig_atomics = smolF.pad_tensors(lig_atomics)
        if "charges" in data:
            lig_charges = smolF.pad_tensors(lig_charges)
        lig_bonds = torch.stack(lig_bonds)

        out = {
            "coords": lig_coords,
            "atomics": lig_atomics,
            "bonds": lig_bonds,
            "mask": atom_mask,
        }
        if "charges" in data:
            out["charges"] = lig_charges
        return out

    def overwrite_pocket(
        self, coords, atomics, bonds, charges, prior, pocket_noise="apo"
    ):
        pocket_mask = prior["pocket_mask"].bool()
        coords_list = []
        type_probs_list = []
        charge_probs_list = []
        bond_probs_list = []
        for idx in range(coords.size(0)):
            if pocket_noise == "fix":
                mol_coords = coords[idx]
                mol_coords[pocket_mask[idx]] = prior["coords"][idx][pocket_mask[idx]]
                coords_list.append(mol_coords)
            mol_type_probs = atomics[idx]
            mol_charge_probs = charges[idx]
            mol_type_probs[pocket_mask[idx]] = prior["atomics"][idx][pocket_mask[idx]]
            mol_charge_probs[pocket_mask[idx]] = prior["charges"][idx][
                pocket_mask[idx]
            ].float()
            type_probs_list.append(mol_type_probs)
            charge_probs_list.append(mol_charge_probs)

            present_indices = pocket_mask[idx].nonzero(as_tuple=True)[0]
            mol_bond_probs = bonds[idx]
            mol_bond_probs[present_indices[:, None], present_indices] = prior["bonds"][
                idx
            ][present_indices[:, None], present_indices]
            bond_probs_list.append(mol_bond_probs)

        if pocket_noise == "fix":
            coords = torch.stack(coords_list)
        type_probs = torch.stack(type_probs_list)
        charge_probs = torch.stack(charge_probs_list)
        bond_probs = torch.stack(bond_probs_list)

        return coords, type_probs, bond_probs, charge_probs


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class MolecularCFM(pl.LightningModule):
    def __init__(
        self,
        gen: MolecularGenerator,
        vocab: Vocabulary,
        lr: float,
        integrator: Integrator,
        coord_scale: float = 1.0,
        type_strategy: str = "ce",
        bond_strategy: str = "ce",
        coord_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        bond_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pocket_noise: str = "random",
        ligand_only: bool = False,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        sampling_strategy: str = "linear",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        dataset_info: DataInfos = None,
        data_path: str = None,
        **kwargs,
    ):
        super().__init__()

        if type_strategy not in ["mse", "ce", "mask"]:
            raise ValueError(f"Unsupported type training strategy '{type_strategy}'")

        if bond_strategy not in ["ce", "mask"]:
            raise ValueError(f"Unsupported bond training strategy '{bond_strategy}'")

        if lr_schedule not in ["constant", "one-cycle", "exponential"]:
            raise ValueError(f"LR scheduler {lr_schedule} not supported.")

        if lr_schedule == "one-cycle" and total_steps is None:
            raise ValueError(
                "total_steps must be provided when using the one-cycle LR scheduler."
            )

        if distill and (type_strategy == "mask" or bond_strategy == "mask"):
            raise ValueError(
                "Distilled training with masking strategy is not supported."
            )

        # Note that warm_up_steps is currently ignored if schedule is one-cycle

        self.gen = gen
        self.vocab = vocab
        self.lr = lr
        self.coord_scale = coord_scale
        self.type_strategy = type_strategy
        self.bond_strategy = bond_strategy
        self.coord_loss_weight = coord_loss_weight
        self.type_loss_weight = type_loss_weight
        self.bond_loss_weight = bond_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.pocket_noise = pocket_noise
        self.ligand_only = ligand_only

        # Anything else passed into kwargs will also be saved
        hparams = {
            "lr": lr,
            "coord_scale": coord_scale,
            "coord_loss_weight": coord_loss_weight,
            "type_loss_weight": type_loss_weight,
            "bond_loss_weight": bond_loss_weight,
            "type_strategy": type_strategy,
            "bond_strategy": bond_strategy,
            "self_condition": self_condition,
            "pocket_noise": pocket_noise,
            "ligand_only": ligand_only,
            "distill": distill,
            "lr_schedule": lr_schedule,
            "sampling_strategy": sampling_strategy,
            "use_ema": use_ema,
            "compile_model": compile_model,
            "warm_up_steps": warm_up_steps,
            "data_path": data_path,
            **gen.hparams,
            **integrator.hparams,
            **kwargs,
        }
        self.save_hyperparameters(hparams)

        builder = MolBuilder(
            vocab, pocket_noise=self.pocket_noise, save_dir=self.hparams.save_dir
        )

        if compile_model:
            self.gen = self._compile_model(gen)

        self.integrator = integrator
        self.builder = builder
        self.dataset_info = dataset_info

        self.molecule_list = []
        self.train_smiles = train_smiles

        if self.train_smiles is not None:
            explicit_hs = self.hparams.dataset in ["qm9", "geom-drugs"]
            if Path(os.path.join(self.hparams.data_path, "train_mols.pkl")).exists():
                print("Loading RDKit training mols...")
                with open(
                    os.path.join(self.hparams.data_path, "train_mols.pkl"), "rb"
                ) as f:
                    train_mols = pickle.load(f)
                print("Done.")
            else:
                print("Creating RDKit mols from training SMILES...")
                train_mols = self.builder.mols_from_smiles(
                    self.train_smiles, explicit_hs=explicit_hs
                )
                train_mols = [mol for mol in train_mols if mol is not None]
                with open(
                    os.path.join(self.hparams.data_path, "train_mols.pkl"), "wb"
                ) as f:
                    pickle.dump(train_mols, f)
                print("Done.")

        self.stability_metrics = None
        self.gen_sb_metrics = None
        self.gen_complex_metrics = None
        self.gen_pocket_metrics = None
        self.gen_plif_recovery = None
        gen_mol_metrics = {
            "validity": Metrics.Validity(),
            "fc-validity": Metrics.Validity(connected=True),
            "uniqueness": Metrics.Uniqueness(),
            "energy-validity": Metrics.EnergyValidity(),
            "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
            "energy": Metrics.AverageEnergy(),
            "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
            "strain": Metrics.AverageStrainEnergy(),
            "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
            "opt-rmsd": Metrics.AverageOptRmsd(),
        }
        gen_dist_metrics = Metrics.DistributionDistance(dataset_info=self.dataset_info)
        self.gen_dist_metrics = MetricCollection(
            {"distribution-distance": gen_dist_metrics}, compute_groups=False
        )

        if self.hparams.dataset in ["qm9", "geom-drugs"]:
            stability_metrics = {
                "atom-stability": Metrics.AtomStability(),
                "molecule-stability": Metrics.MoleculeStability(),
            }
            self.stability_metrics = MetricCollection(
                stability_metrics, compute_groups=False
            )

        # elif self.hparams.dataset == "plinder" or self.hparams.dataset == "crossdocked":
        #     config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))
        #     # mol_list is a list of RDKit molecules
        #     gen_mol_metrics.update({"genbench3d": Metrics.GenBench3DValidity(config)})
        #     gen_complex_metrics = {
        #         "sb_genbench3d": Metrics.GenBench3DSB(config),
        #         "posebusters": Metrics.PoseBustersValidity(),
        #         "ifp_similarity": Metrics.IFPSimilarity(),
        #     }
        #     gen_pocket_metrics = {"pocket_validity": Metrics.PocketValidity()}
        #     self.gen_complex_metrics = MetricCollection(
        #         gen_complex_metrics, compute_groups=False
        #     )
        #     self.gen_pocket_metrics = MetricCollection(
        #         gen_pocket_metrics, compute_groups=False
        #     )
        #     gen_plif_recovery = {"interaction_recovery": Metrics.InteractionRecovery(plifs_dict_ground_truth, ground_truth_plifs_path)}
        #     self.gen_plif_recovery = MetricCollection(gen_plif_recovery, compute_groups=False)

        if self.train_smiles is not None:
            print("Initialising novelty metric...")
            gen_mol_metrics["novelty"] = Metrics.Novelty(train_mols)
            print("Novelty metric complete.")

        self.gen_mol_metrics = MetricCollection(gen_mol_metrics, compute_groups=False)

        if pairwise_metrics:
            pair_metrics = {
                "mol-accuracy": Metrics.MolecularAccuracy(),
                "pair-rmsd": Metrics.MolecularPairRMSD(),
            }
            self.pair_metrics = MetricCollection(pair_metrics, compute_groups=False)

        self._init_params()

    def forward(self, batch, t, training=False, cond_batch=None):
        """Predict molecular coordinates and atom types

        Args:
            batch (dict[str, Tensor]): Batched pointcloud data
            t (torch.Tensor): Interpolation times between 0 and 1, shape [batch_size]
            training (bool): Whether to run forward in training mode
            cond_batch (dict[str, Tensor]): Predictions from previous step, if we are using self conditioning

        Returns:
            (predicted coordinates, atom type logits (unnormalised probabilities))
            Both torch.Tensor, shapes [batch_size, num_atoms, 3] and [batch_size, num atoms, vocab_size]
        """

        coords = batch["coords"]
        atom_types = batch["atomics"]
        bonds = batch["bonds"]
        mask = batch["mask"].bool()

        # Prepare invariant atom features
        times = t.view(-1, 1, 1).expand(-1, coords.size(1), -1)

        # add whether atom is in the ligand or the pocket
        lig_mask, pocket_atoms, pocket_res = None, None, None
        if "lig_mask" in batch:
            lig_mask = batch["lig_mask"].bool()
            is_lig_atom = (
                torch.zeros(atom_types.shape[:2]).unsqueeze(-1).to(atom_types.device)
            )
            is_lig_atom[lig_mask] = 1
            features = torch.cat((is_lig_atom, times, atom_types), dim=2)
            pocket_atoms = batch["atom_names"]
            pocket_res = batch["res_names"]
        else:
            features = torch.cat((times, atom_types), dim=2)

        if cond_batch is not None:
            out = self.gen(
                coords,
                inv_feats=features,
                pocket_atoms=pocket_atoms,
                pocket_res=pocket_res,
                edge_feats=bonds,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                atom_mask=mask,
                lig_mask=lig_mask,
                atoms=atom_types,
                bonds=bonds,
                pocket_mask=batch["pocket_mask"],
            )
        else:
            out = self.gen(
                coords,
                features,
                pocket_atoms=pocket_atoms,
                pocket_res=pocket_res,
                edge_feats=bonds,
                atom_mask=mask,
                lig_mask=lig_mask,
                atoms=atom_types,
                bonds=bonds,
                pocket_mask=batch["pocket_mask"],
            )

        return out

    def training_step(self, batch, b_idx):
        _, data, interpolated, times = batch

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:

            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"]),
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_coords, cond_types, cond_bonds, _, mask = self(
                        interpolated, times, training=True, cond_batch=cond_batch
                    )
                    cond_batch = {
                        "coords": cond_coords,
                        "atomics": F.softmax(cond_types, dim=-1),
                        "bonds": F.softmax(cond_bonds, dim=-1),
                    }
                    if self.ligand_only:
                        cond_batch = self.builder.add_ligand_to_pocket(
                            lig_data=cond_batch,
                            lig_mask=mask.bool(),
                            complex_data=interpolated,
                            add_charges=False,
                            add_pocket_info=False,
                        )

        coords, types, bonds, charges, mask = self(
            interpolated, times, training=True, cond_batch=cond_batch
        )
        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        # self.builder.tensors_to_xyz(
        #     prior=batch[0],
        #     interpolated=interpolated,
        #     data=data,
        #     predicted=predicted,
        #     coord_scale=self.hparams.coord_scale,
        #     idx=0,
        #     save_dir=f"{self.hparams.save_dir}/tmp",
        # )
        # import pdb

        # pdb.set_trace()

        losses = self._loss(data, interpolated, predicted)
        loss = sum(list(losses.values()))

        for name, loss_val in losses.items():
            self.log(
                f"train-{name}",
                loss_val,
                prog_bar=True,
                on_step=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            "train-loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True
        )

        # loss.backward()
        # names = []
        # for name, param in self.gen.named_parameters():
        #     if param.grad is None:
        #         names.append(name)
        # import pdb

        # pdb.set_trace()

        return loss

    def on_after_backward(self):
        for name, param in self.named_parameters():
            if param.grad is None:
                print(name)

    def validation_step(self, batch, b_idx):
        prior, data, interpolated, times = batch

        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)

        if self.hparams.dataset in ["qm9", "geom-drugs"]:
            gen_mols = self._generate_mols(gen_batch)
            stabilities = self._generate_stabilities(gen_batch)
            self.stability_metrics.update(stabilities)

        elif (
            self.hparams.dataset == "plinder"
            or self.hparams.dataset == "crossdocked"
            or self.hparams.dataset == "kinodata"
        ):
            gen_mols = (
                self._generate_mols(gen_batch)
                if self.ligand_only
                else self._generate_ligs(gen_batch, mask=prior["lig_mask"].bool())
            )
            if not self.ligand_only:
                ref_mols = self._generate_ligs(
                    data, mask=data["lig_mask"].bool(), scale=self.coord_scale
                )
                # ref_pockets = self._generate_pockets(
                #     data, mask=data["pocket_mask"].bool()
                # )
                ref_pdbs = self._generate_pdbs(data, stage="val_ref")
                if self.gen_complex_metrics is not None:
                    gen_pdbs = self._generate_pdbs(
                        data,
                        coords=gen_batch["coords"],
                        stage="val_gen",
                    )
                    self.gen_complex_metrics.update(gen_mols, ref_mols, gen_pdbs)

                    if self.gen_pocket_metrics is not None:
                        self.gen_pocket_metrics.update(ref_pdbs, gen_pdbs)

        self.gen_mol_metrics.update(gen_mols)
        self.gen_dist_metrics.update(gen_mols)

        # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
        if self.pairwise_metrics and not (
            self.hparams.dataset == "plinder"
            or self.hparams.dataset == "crossdocked"
            or self.hparams.dataset == "kinodata"
        ):  # TO DO: Remove the dataset condition at some point - might also be interesting for SBDD (must be adapted though, so for now commented out)
            gen_interp_steps = max(
                1, int((1 - times[0].item()) * self.integrator.steps)
            )
            gen_interp_batch = self._generate(interpolated, gen_interp_steps)
            gen_interp_mols = self._generate_mols(gen_interp_batch)
            data_mols = self._generate_mols(data, scale=self.coord_scale)
            self.pair_metrics.update(gen_interp_mols, data_mols)

    def on_validation_epoch_end(self):
        stability_metrics_results, pair_metrics_results, gen_complex_metrics_results = (
            {},
            {},
            {},
        )
        if self.hparams.dataset in ["qm9", "geom-drugs"]:
            stability_metrics_results = self.stability_metrics.compute()
            pair_metrics_results = (
                self.pair_metrics.compute() if self.pairwise_metrics else {}
            )

        gen_metrics_results = self.gen_mol_metrics.compute()
        gen_dist_results = self.gen_dist_metrics.compute()

        if self.gen_complex_metrics is not None and not self.ligand_only:
            gen_complex_metrics_results = self.gen_complex_metrics.compute()
        if self.gen_pocket_metrics is not None and not self.ligand_only:
            gen_pocket_metrics_results = self.gen_pocket_metrics.compute()

        metrics = {
            **stability_metrics_results,
            **gen_metrics_results,
            **pair_metrics_results,
            **gen_dist_results,
        }
        if self.gen_complex_metrics is not None and not self.ligand_only:
            metrics = {
                **metrics,
                **gen_complex_metrics_results,
                **gen_pocket_metrics_results,
            }

        for metric, value in metrics.items():
            progbar = True if metric == "fc-validity" else False
            if isinstance(value, dict):
                for k, v in value.items():
                    self.log(
                        f"val-{k}",
                        v.to(self.device),
                        on_epoch=True,
                        logger=True,
                        prog_bar=progbar,
                        sync_dist=True,
                    )
            else:
                self.log(
                    f"val-{metric}",
                    value.to(self.device),
                    on_epoch=True,
                    logger=True,
                    prog_bar=progbar,
                    sync_dist=True,
                )

        self.gen_mol_metrics.reset()
        self.gen_dist_metrics.reset()

        if self.stability_metrics is not None:
            self.stability_metrics.reset()

        if self.gen_complex_metrics is not None and not self.ligand_only:
            self.gen_complex_metrics.reset()

        if self.gen_pocket_metrics is not None and not self.ligand_only:
            self.gen_pocket_metrics.reset()

        if self.pairwise_metrics:
            self.pair_metrics.reset()

    def run_eval(self, prior):
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        if (
            self.hparams.dataset == "plinder"
            or self.hparams.dataset == "crossdocked"
            or self.hparams.dataset == "kinodata"
        ):
            gen_mols = self._generate_ligs(gen_batch, mask=prior["lig_mask"].bool())
        else:
            gen_mols = self._generate_mols(gen_batch)
        gen_mols = [Molecule(mol, device=self.device) for mol in gen_mols]
        return gen_mols

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        prior, _, _, _ = batch
        gen_batch = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        gen_mols = self._generate_mols(gen_batch)
        return gen_mols

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.gen.parameters(),
            lr=self.lr,
            amsgrad=True,
            foreach=True,
            weight_decay=1e-12,
        )

        if self.lr_schedule == "constant":
            warm_up_steps = 0 if self.warm_up_steps is None else self.warm_up_steps
            scheduler = LinearLR(opt, start_factor=1e-2, total_iters=warm_up_steps)

        # TODO could use warm_up_steps to shift peak of one cycle
        elif self.lr_schedule == "one-cycle":
            scheduler = OneCycleLR(
                opt, max_lr=self.lr, total_steps=self.total_steps, pct_start=0.3
            )
        elif self.lr_schedule == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.998)

        else:
            raise ValueError(f"LR schedule {self.lr_schedule} is not supported.")

        if self.lr_schedule == "constant":
            config = {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
            return config
        else:
            scheduler = {
                "scheduler": scheduler,
                "interval": "epoch",
                # "frequency": self.hparams["lr_frequency"],
                # "monitor": self.validity,
                "strict": False,
            }
        return [opt], [scheduler]

    def _compile_model(self, model):
        return torch.compile(
            model, dynamic=False, fullgraph=True, mode="reduce-overhead"
        )

    def _loss(self, data, interpolated, predicted):
        pred_coords = predicted["coords"]

        if self.ligand_only:
            data = self.builder.extract_ligand_from_complex(data)
            interpolated = self.builder.extract_ligand_from_complex(interpolated)

        coords = data["coords"]
        mask = data["mask"]

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")

        coord_loss_pocket, coord_loss_lig = None, None
        if (
            self.hparams.dataset in ["plinder", "crossdocked", "kinodata"]
            and not self.ligand_only
        ):
            lig_mask = data["lig_mask"].bool()
            num_lig_atoms = lig_mask.sum(-1)
            pocket_mask = data["pocket_mask"].bool()
            num_pocket_atoms = pocket_mask.sum(-1)
            coord_loss = (coord_loss * mask.unsqueeze(-1)).mean(-1)
            coord_loss_lig = (coord_loss * lig_mask).sum(-1) / num_lig_atoms
            coord_loss_pocket = (coord_loss * pocket_mask).sum(-1) / num_pocket_atoms
            coord_loss_lig = coord_loss_lig.mean() * self.coord_loss_weight
            coord_loss_pocket = coord_loss_pocket.mean() * self.coord_loss_weight
        else:
            n_atoms = mask.unsqueeze(-1).sum(dim=(1, 2))
            coord_loss = (coord_loss * mask.unsqueeze(-1)).sum(
                dim=(1, 2)
            ) / n_atoms  # TODO divide loss per pocket/ligand atom by number of pocket/ligand atoms
            coord_loss = coord_loss.mean() * self.coord_loss_weight

        if (
            self.hparams.dataset in ["plinder", "crossdocked", "kinodata"]
            and not self.ligand_only
        ):
            type_loss_lig, type_loss_pocket = self._type_loss(
                data, interpolated, predicted
            )
            bond_loss_lig, bond_loss_pocket = self._bond_loss(
                data, interpolated, predicted
            )
            charge_loss_lig, charge_loss_pocket = self._charge_loss(data, predicted)
            pocket_scale = 0.0 if self.pocket_noise == "fix" else 1.0
            losses = {
                "coord-loss-lig": (
                    coord_loss_lig if self.coord_scale == 1.0 else coord_loss_lig * 6
                ),
                "type-loss-lig": (
                    type_loss_lig if self.coord_scale == 1.0 else type_loss_lig
                ),
                "bond-loss-lig": (
                    bond_loss_lig * 3 if self.coord_scale == 1.0 else bond_loss_lig * 3
                ),
                "charge-loss-lig": charge_loss_lig,
                "coord-loss-pocket": coord_loss_pocket * pocket_scale,
                "bond-loss-pocket": bond_loss_pocket * pocket_scale,
                "type-loss-pocket": type_loss_pocket * pocket_scale,
                "charge-loss-pocket": charge_loss_pocket * pocket_scale,
            }
        else:
            type_loss = self._type_loss(data, interpolated, predicted)
            bond_loss = self._bond_loss(data, interpolated, predicted)
            charge_loss = self._charge_loss(data, predicted)

            losses = {
                "coord-loss": coord_loss,
                "type-loss": type_loss,
                "bond-loss": bond_loss,
                "charge-loss": charge_loss,
            }
        return losses

    def _type_loss(self, data, interpolated, predicted, eps=1e-3):

        pred_logits = predicted["atomics"]

        atomics_dist = data["atomics"]
        mask = data["mask"].unsqueeze(2)

        batch_size, num_atoms, _ = pred_logits.size()

        if self.type_strategy == "mse":
            type_loss = F.mse_loss(pred_logits, atomics_dist, reduction="none")
        else:
            atomics = torch.argmax(atomics_dist, dim=-1).flatten(0, 1)
            type_loss = F.cross_entropy(
                pred_logits.flatten(0, 1), atomics, reduction="none"
            )
            type_loss = type_loss.unflatten(0, (batch_size, num_atoms)).unsqueeze(2)

        n_atoms = mask.sum(dim=(1, 2)) + eps

        # If we are training with masking, only compute the loss on masked types
        if self.type_strategy == "mask":
            at = interpolated["atomics"]
            masked_types = torch.argmax(at, dim=-1) == self.type_mask_index
            n_atoms = masked_types.sum(dim=-1) + eps
            type_loss = type_loss * masked_types.float().unsqueeze(-1)

        if (
            self.hparams.dataset in ["plinder", "crossdocked", "kinodata"]
            and not self.ligand_only
        ):
            if len(type_loss.size()) == 3:
                type_loss = type_loss.squeeze(-1)
            lig_mask = data["lig_mask"].bool()
            num_lig_atoms = lig_mask.sum(-1)
            pocket_mask = data["pocket_mask"].bool()
            num_pocket_atoms = pocket_mask.sum(-1)
            type_loss = type_loss * mask.squeeze(-1)
            type_loss_lig = (type_loss * lig_mask).sum(-1) / num_lig_atoms
            type_loss_pocket = (type_loss * pocket_mask).sum(-1) / num_pocket_atoms
            type_loss_lig = type_loss_lig.mean() * self.type_loss_weight
            type_loss_pocket = type_loss_pocket.mean() * self.type_loss_weight
            return type_loss_lig, type_loss_pocket

        else:
            type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
            type_loss = type_loss.mean() * self.type_loss_weight

            return type_loss

    def _bond_loss(self, data, interpolated, predicted, eps=1e-3):
        pred_logits = predicted["bonds"]

        mask = data["mask"]
        bonds = torch.argmax(data["bonds"], dim=-1)

        batch_size, num_atoms, _, _ = pred_logits.size()

        bond_loss = F.cross_entropy(
            pred_logits.flatten(0, 2), bonds.flatten(0, 2), reduction="none"
        )
        bond_loss = bond_loss.unflatten(0, (batch_size, num_atoms, num_atoms))

        adj_matrix = smolF.adj_from_node_mask(mask, self_connect=True)
        n_bonds = adj_matrix.sum(dim=(1, 2)) + eps

        # Only compute loss on masked bonds if we are training with masking strategy
        if self.bond_strategy == "mask":
            bt = interpolated["bonds"]
            masked_bonds = torch.argmax(bt, dim=-1) == self.bond_mask_index
            n_bonds = masked_bonds.sum(dim=(1, 2)) + eps
            bond_loss = bond_loss * masked_bonds.float()

        if (
            self.hparams.dataset in ["plinder", "crossdocked", "kinodata"]
            and not self.ligand_only
        ):
            lig_mask = data["lig_mask"]
            pocket_mask = data["pocket_mask"]
            lig_adj_matrix = smolF.adj_from_node_mask(lig_mask, self_connect=True)
            pocket_adj_matrix = smolF.adj_from_node_mask(pocket_mask, self_connect=True)
            num_lig_bonds = (lig_adj_matrix.sum(dim=(1, 2)) + eps).int()
            num_pocket_bonds = (pocket_adj_matrix.sum(dim=(1, 2)) + eps).int()

            bond_loss_lig = (bond_loss * lig_adj_matrix).sum(dim=(1, 2)) / num_lig_bonds
            bond_loss_lig = bond_loss_lig.mean() * self.bond_loss_weight
            bond_loss_pocket = (bond_loss * pocket_adj_matrix).sum(
                dim=(1, 2)
            ) / num_pocket_bonds
            bond_loss_pocket = bond_loss_pocket.mean() * self.bond_loss_weight
            return bond_loss_lig, bond_loss_pocket
        else:
            bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
            bond_loss = bond_loss.mean() * self.bond_loss_weight
            return bond_loss

    def _charge_loss(self, data, predicted, eps=1e-3):
        pred_logits = predicted["charges"]

        charges = data["charges"]
        mask = data["mask"]

        batch_size, num_atoms, _ = pred_logits.size()

        charges = torch.argmax(charges, dim=-1).flatten(0, 1)
        charge_loss = F.cross_entropy(
            pred_logits.flatten(0, 1), charges, reduction="none"
        )
        charge_loss = charge_loss.unflatten(0, (batch_size, num_atoms))

        n_atoms = mask.sum(dim=1) + eps

        if (
            self.hparams.dataset in ["plinder", "crossdocked", "kinodata"]
            and not self.ligand_only
        ):
            lig_mask = data["lig_mask"].bool()
            num_lig_atoms = lig_mask.sum(-1)
            pocket_mask = data["pocket_mask"].bool()
            num_pocket_atoms = pocket_mask.sum(-1)
            charge_loss_lig = (charge_loss * lig_mask).sum(-1) / num_lig_atoms
            charge_loss_pocket = (charge_loss * pocket_mask).sum(-1) / num_pocket_atoms
            charge_loss_lig = charge_loss_lig.mean() * self.charge_loss_weight
            charge_loss_pocket = charge_loss_pocket.mean() * self.charge_loss_weight
            return charge_loss_lig, charge_loss_pocket
        else:
            charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
            charge_loss = charge_loss.mean() * self.charge_loss_weight
            return charge_loss

    def _generate(self, prior, steps, strategy="linear", iter=0, save_traj=False):

        if strategy == "linear":
            time_points = np.linspace(0, 1, steps + 1).tolist()

        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 1.0, steps + 1)).tolist()
            time_points.reverse()
        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        times = torch.zeros(prior["coords"].size(0), device=self.device)
        curr = {
            k: v.clone() if torch.is_tensor(v) else v.copy() for k, v in prior.items()
        }

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]

        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        with torch.no_grad():
            for i, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None
                coords, type_logits, bond_logits, charge_logits, mask = self(
                    curr, times, training=False, cond_batch=cond
                )

                type_probs = F.softmax(type_logits, dim=-1)
                bond_probs = F.softmax(bond_logits, dim=-1)
                charge_probs = F.softmax(charge_logits, dim=-1)

                if (
                    self.pocket_noise in ["fix", "apo"] and not self.ligand_only
                ):  # overwrite the pocket atom and bond type predictions with the holo data
                    coords, type_probs, bond_probs, charge_probs = (
                        self.builder.overwrite_pocket(
                            coords,
                            type_probs,
                            bond_probs,
                            charge_probs,
                            prior,
                            pocket_noise=self.pocket_noise,
                        )
                    )

                cond_batch = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                }
                predicted = {
                    "coords": coords,
                    "atomics": type_probs,
                    "bonds": bond_probs,
                    "charges": charge_probs,
                    "mask": mask,
                }

                if self.pocket_noise in ["fix", "apo"] and not self.ligand_only:
                    pocket_mask = prior["pocket_mask"].bool()
                    assert torch.equal(
                        curr["atomics"][pocket_mask].argmax(-1),
                        prior["atomics"][pocket_mask].argmax(-1),
                    ), "Predicted pocket atoms must be equal the holo atoms"

                if self.ligand_only:
                    # extract ligand from complex to do interpolation only on ligand

                    curr = self.builder.extract_ligand_from_complex(curr)
                    _prior = self.builder.extract_ligand_from_complex(prior)

                    curr = self.integrator.step(
                        curr, predicted, _prior, times, step_size
                    )
                    curr = self.builder.add_ligand_to_pocket(
                        lig_data=curr,
                        lig_mask=mask.bool(),
                        complex_data=prior,
                        add_charges=False,
                        add_pocket_info=True,
                    )
                    if self.self_condition:
                        cond_batch = self.builder.add_ligand_to_pocket(
                            lig_data=cond_batch,
                            lig_mask=mask.bool(),
                            complex_data=prior,
                            add_charges=False,
                            add_pocket_info=False,
                        )
                else:
                    curr = self.integrator.step(
                        curr, predicted, prior, times, step_size
                    )

                if save_traj:
                    self.builder.write_xyz_file_from_batch(
                        data=predicted,
                        coord_scale=self.coord_scale,
                        path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
                        t=i,
                    )
                    self.builder.write_xyz_file_from_batch(
                        data=curr,
                        coord_scale=self.coord_scale,
                        path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
                        t=i,
                    )

                times = times + step_size

        if save_traj:
            self.builder.write_trajectory_as_xyz(
                num_mols=coords.size(0),
                file_path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
                save_path=os.path.join(
                    self.hparams.save_dir, "trajectories_pred", f"traj_{iter}"
                ),
                remove_intermediate_files=True,
            )
            self.builder.write_trajectory_as_xyz(
                num_mols=coords.size(0),
                file_path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
                save_path=os.path.join(
                    self.hparams.save_dir, "trajectories_interp", f"traj_{iter}"
                ),
                remove_intermediate_files=True,
            )

        predicted["coords"] = predicted["coords"] * self.coord_scale
        return predicted

    def _generate_mols(self, generated, scale=1.0, sanitise=True):
        coords = generated["coords"] * scale
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]

        mols = self.builder.mols_from_tensors(
            coords,
            atom_dists,
            masks,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def _generate_ligs(self, generated, mask, scale=1.0, sanitise=True):

        coords = generated["coords"] * scale
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]

        mols = self.builder.ligs_from_complex(
            coords,
            mask=mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def _generate_pockets(self, data, mask):

        coords = data["coords"]
        atoms = data["atomics"]
        bonds = data["bonds"]
        charges = data["charges"]
        pockets = self.builder.pockets_from_complex(
            coords,
            pocket_mask=mask,
            atom_dists=atoms,
            bond_dists=bonds,
            charge_dists=charges,
        )
        return pockets

    def _generate_pdbs(self, data, coords=None, iter="", stage="ref_val"):
        """
        Generate PDB files from the PocketComplex data.
        If (predicted) coords are provided, the PDB files will be generated using these coordinates, otherwise the (CoM aligned and potentially normalized) coordinates from the data will be used.
        """
        pdb_path = Path(self.hparams.data_path) / f"{stage}_pdbs"
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            pdb_path / (_complex.metadata["system_id"] + f"{iter}.pdb")
            for _complex in data["complex"]
        ]
        coords = data["coords"] * self.coord_scale if coords is None else coords
        _ = [
            _complex.apo.set_coords(coord[mask.bool()].cpu().numpy()).write_pdb(
                pdb_file
            )
            for _complex, coord, mask, pdb_file in zip(
                data["complex"], coords, data["pocket_mask"], pdb_files
            )
        ]
        return pdb_files

    def _generate_stabilities(self, generated):
        coords = generated["coords"]
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]
        masks = generated["mask"]
        stabilities = self.builder.mol_stabilities(
            coords, atom_dists, masks, bond_dists, charge_dists
        )
        return stabilities

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
