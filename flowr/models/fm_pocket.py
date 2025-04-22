import os
import pickle
import shutil
from collections import defaultdict
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
from flowr.util.molrepr import GeometricMol
from flowr.util.tokeniser import Vocabulary

_T = torch.Tensor
_BatchT = dict[str, _T]


def create_list_defaultdict():
    return defaultdict(list)


class Integrator:
    def __init__(
        self,
        steps,
        coord_noise_std=0.0,
        type_strategy="uniform-sample",
        bond_strategy="uniform-sample",
        pocket_noise=None,
        ligand_only=False,
        cat_noise_level=0,
        corrector_sch_a=0.25,
        corrector_sch_b=0.25,
        corrector_iter_weight=10.0,
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
        self.corrector_sch_a = corrector_sch_a
        self.corrector_sch_b = corrector_sch_b
        self.corrector_iter_weight = corrector_iter_weight
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
            "corrector-sch-a": self.corrector_sch_a,
            "corrector-sch-b": self.corrector_sch_b,
            "corrector-iter-weight": self.corrector_iter_weight,
        }

    def step(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        times: list,
        step_size: float,
    ) -> _BatchT:

        device = curr["coords"].device
        vocab_size = predicted["atomics"].size(-1)
        n_bonds = predicted["bonds"].size(-1)
        lig_times_cont = times[0]
        lig_times_disc = times[1]
        # pocket_times = times[2]
        interaction_times = times[-1]

        # *** Coord update step ***
        coord_velocity = (predicted["coords"] - curr["coords"]) / (
            1 - lig_times_cont.view(-1, 1, 1)
        )
        coord_velocity += torch.randn_like(coord_velocity) * self.coord_noise_std
        coords = curr["coords"] + (step_size * coord_velocity)
        coords = coords * prior["mask"].unsqueeze(-1)

        # *** Atom type update step ***
        if self.type_strategy == "linear":
            one_hots = torch.eye(vocab_size, device=device).unsqueeze(0).unsqueeze(0)
            type_velocity = one_hots - prior["atomics"].unsqueeze(-1)
            type_velocity = (type_velocity * predicted["atomics"].unsqueeze(-2)).sum(-1)
            atomics = curr["atomics"] + (step_size * type_velocity)

        # Masking strategy from Discrete Flow Models paper (https://arxiv.org/abs/2402.04997)
        elif self.type_strategy == "mask":
            atomics = self._mask_sampling_step(
                curr["atomics"],
                predicted["atomics"],
                lig_times_disc,
                self.type_mask_index,
                step_size,
            )

        # Uniform sampling strategy from Discrete Flow Models paper
        elif self.type_strategy == "uniform-sample":
            atomics = self._uniform_sample_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )

        # Uniform sampling from discrete flow models paper
        elif self.type_strategy == "velocity-sample":
            atomics = self._velocity_sample_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )

        # *** Bond update step ***
        if self.bond_strategy == "linear":
            one_hots = torch.eye(n_bonds, device=device).view(1, 1, 1, n_bonds, n_bonds)
            bond_velocity = one_hots - prior["bonds"].unsqueeze(-1)
            bond_velocity = (bond_velocity * predicted["bonds"].unsqueeze(-2)).sum(-1)
            bonds = curr["bonds"] + (step_size * bond_velocity)

        elif self.bond_strategy == "mask":
            bonds = self._mask_sampling_step(
                curr["bonds"],
                predicted["bonds"],
                lig_times_disc,
                self.bond_mask_index,
                step_size,
            )

        elif self.bond_strategy == "uniform-sample":
            bonds = self._uniform_sample_step(
                curr["bonds"], predicted["bonds"], lig_times_disc, step_size
            )

        # Uniform sampling from discrete flow models paper
        elif self.bond_strategy == "velocity-sample":
            bonds = self._velocity_sample_step(
                curr["bonds"], predicted["bonds"], lig_times_disc, step_size
            )

        interactions = None
        if "interactions" in predicted:
            interactions = self._uniform_sample_step(
                curr["interactions"],
                predicted["interactions"],
                interaction_times,
                step_size,
            )

        updated = {
            "coords": coords,
            "atomics": atomics.float(),
            "bonds": bonds.float(),
            "mask": curr["mask"],
        }
        if interactions is not None:
            updated["interactions"] = interactions.float()

        return updated

    def corrector_iter(
        self,
        curr: _BatchT,
        predicted: _BatchT,
        prior: _BatchT,
        times: list,
        step_size: float,
    ) -> _BatchT:
        lig_times_disc = times[1]

        if self.type_strategy == "velocity-sample":
            atomics = self._corrector_iter_step(
                curr["atomics"], predicted["atomics"], lig_times_disc, step_size
            )
        else:
            raise ValueError(
                "Type strategy must be velocity-sample for corrector iterations"
            )

        if self.bond_strategy == "velocity-sample":
            bonds = self._corrector_iter_step(
                curr["bonds"], predicted["bonds"], lig_times_disc, step_size
            )
        else:
            raise ValueError(
                "Bond strategy must be velocity-sample for corrector iterations"
            )

        updated = {
            "coords": predicted["coords"],
            "atomics": atomics.float(),
            "bonds": bonds.float(),
            "mask": curr["mask"],
        }
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

    def _velocity_sample_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        beta_t = torch.pow(times, self.corrector_sch_a) * torch.pow(
            1 - times, self.corrector_sch_b
        )
        beta_t = self.cat_noise_level * beta_t
        alpha_t = beta_t + 1

        # Curr dist should be one-hot
        forward_vel = (pred_dist - curr_dist) * (1 / (1 - times))

        # Assume uniform dist for prior
        backward_vel = (curr_dist - (1 / n_categories)) * (1 / times)

        prob_vel = (alpha_t * forward_vel) - (beta_t * backward_vel)
        step_dist = curr_dist + (step_size * prob_vel)
        step_dist = step_dist.clamp(min=0.0)

        samples = torch.distributions.Categorical(step_dist).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _corrector_iter_step(self, curr_dist, pred_dist, t, step_size):
        n_categories = pred_dist.size(-1)

        ones = [1] * (len(pred_dist.shape) - 1)
        times = t.view(-1, *ones).clamp(min=self.eps, max=1.0 - self.eps)

        # Curr dist should be one-hot
        forward_vel = (pred_dist - curr_dist) * (1 / (1 - times))

        # NOTE Assumes uniform dist for prior
        backward_vel = (curr_dist - (1 / n_categories)) * (1 / times)

        prob_vel = (self.corrector_iter_weight * forward_vel) - (
            self.corrector_iter_weight * backward_vel
        )
        step_dist = curr_dist + (step_size * prob_vel)
        step_dist = step_dist.clamp(min=0.0)

        samples = torch.distributions.Categorical(step_dist).sample()
        return smolF.one_hot_encode_tensor(samples, n_categories)

    def _check_cat_sampling_strategy(self, strategy, mask_index, name):
        if strategy not in ["linear", "mask", "uniform-sample", "velocity-sample"]:
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
            mask = predicted["mask"]
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
        pred_mols,
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
        for i in range(len(pred_mols)):
            if smolRD.mol_is_valid(pred_mols[i], connected=True):
                files = sorted(
                    glob(os.path.join(file_path, f"graph_{i}/latent_*.xyz")),
                    key=get_key,
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
        pocket_data,
    ):
        coords, atomics = [], []

        pocket_coords = pocket_data["coords"].clone()
        pocket_atomics = pocket_data["atomics"].clone()
        pocket_mask = pocket_data["mask"].bool()

        lig_coords, lig_atomics, lig_mask = (
            lig_data["coords"],
            lig_data["atomics"],
            lig_data["mask"],
        )
        lig_mask = lig_mask.bool()

        for idx in range(lig_coords.size(0)):
            coords.append(
                torch.cat(
                    [
                        pocket_coords[idx, pocket_mask[idx], :],
                        lig_coords[idx, lig_mask[idx], :],
                    ],
                    dim=0,
                )
            )
            atomics.append(
                torch.cat(
                    [
                        pocket_atomics[idx, pocket_mask[idx], :],
                        lig_atomics[idx, lig_mask[idx], :],
                    ],
                    dim=0,
                )
            )

        mask = (
            smolF.pad_tensors([torch.ones(len(coord)) for coord in coords])
            .to(lig_mask.device)
            .int()
        )
        coords = smolF.pad_tensors(coords)
        atomics = smolF.pad_tensors(atomics)
        out = {
            "coords": coords,
            "atomics": atomics,
            "mask": mask,
        }
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

    def extract_pocket_from_complex(self, data):
        coords = data["coords"]
        atomics = data["atomics"]
        bonds = data["bonds"]
        charges = data["charges"]
        pocket_mask = data["pocket_mask"].bool()
        max_atoms = pocket_mask.sum(dim=1).max().item()

        pocket_coords = []
        pocket_atomics = []
        pocket_charges = []
        pocket_bonds = []
        pocket_atoms = []
        pocket_res_names = []
        for i in range(coords.size(0)):
            pocket_coords.append(coords[i, pocket_mask[i], :])
            pocket_atomics.append(atomics[i, pocket_mask[i], :])
            pocket_atoms.append(data["atom_names"][i][pocket_mask[i]])
            pocket_res_names.append(data["res_names"][i][pocket_mask[i]])
            pocket_charges.append(charges[i, pocket_mask[i], :])
            num_atoms = pocket_mask[i].sum().item()
            bond_probs = torch.zeros(max_atoms, max_atoms, bonds.shape[-1]).to(
                pocket_mask.device
            )
            bond_indices = pocket_mask[i].nonzero(as_tuple=True)[0]
            bond_probs[:num_atoms, :num_atoms, :] = bonds[i][
                bond_indices[:, None], bond_indices
            ]
            pocket_bonds.append(bond_probs)
        atom_mask = (
            smolF.pad_tensors([torch.ones(len(coords)) for coords in pocket_coords])
            .to(pocket_mask.device)
            .int()
        )
        pocket_coords = smolF.pad_tensors(pocket_coords)
        pocket_atomics = smolF.pad_tensors(pocket_atomics)
        pocket_atoms = smolF.pad_tensors(pocket_atoms)
        pocket_res_names = smolF.pad_tensors(pocket_res_names)
        pocket_charges = smolF.pad_tensors(pocket_charges)
        pocket_bonds = torch.stack(pocket_bonds)

        out = {
            "coords": pocket_coords,
            "atomics": pocket_atomics,
            "atom_names": pocket_atoms,
            "res_names": pocket_res_names,
            "bonds": pocket_bonds,
            "charges": pocket_charges,
            "mask": atom_mask,
        }
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

    def _inpaint_times(self, times, mask):
        times = times.masked_fill(mask, 0.999)
        return times

    def inpaint_molecule(
        self,
        data: dict,
        prediction: dict,
        pocket_mask: torch.Tensor,
        keep_interactions: bool = False,
    ) -> dict:
        """
        Vectorized inpainting based on the interactions.
        Returns:
        dict: Updated prediction dictionary with inpainted coordinates, atomics, and bonds.
        """
        # Unpack prior
        coords = data["coords"]  # (B, N_l, C)
        atomics = data["atomics"]  # (B, N_l, d)
        bonds = data["bonds"]  # (B, N_l, N_l, n_bonds)
        fragment_mask = data["fragment_mask"].bool()  # (B, N_l)

        # Unpack predicted
        pred_coords = prediction["coords"]  # (B, N_l, C)
        pred_atomics = prediction["atomics"]  # (B, N_l, d)
        pred_bonds = prediction["bonds"]  # (B, N_l, N_l, n_bonds)
        lig_mask = data["mask"].bool()  # (B, N_l)

        # Inpainting mask
        inpaint_mask = fragment_mask & lig_mask  # (B, N_l)

        # Overwrite coordinates and atomics where interactions are present
        pred_coords[inpaint_mask, :] = coords[inpaint_mask, :]
        pred_atomics[inpaint_mask, :] = atomics[inpaint_mask, :]

        # Overwrite bonds with a pairwise fixed mask:
        inpaint_mask = inpaint_mask.unsqueeze(2) & inpaint_mask.unsqueeze(1)
        bond_mask = torch.argmax(bonds, dim=-1) != 0
        fixed_mask = inpaint_mask & bond_mask
        pred_bonds[fixed_mask] = bonds[fixed_mask]
        # pred_bonds = torch.where(fixed_mask, bonds, pred_bonds)

        out = {
            "coords": pred_coords,
            "atomics": pred_atomics,
            "bonds": pred_bonds,
            "mask": data["mask"],
            "fragment_mask": fragment_mask,
        }
        if "charges" in prediction:
            out["charges"] = prediction["charges"]
        if keep_interactions and "interactions" in prediction:
            out["interactions"] = prediction["interactions"]

        return out

    def inpaint_interactions(
        self,
        data: dict,
        prediction: dict,
        pocket_mask: torch.Tensor,
        keep_interactions: bool = False,
    ) -> dict:
        """
        Vectorized inpainting based on the interactions.
        Returns:
        dict: Updated prediction dictionary with inpainted coordinates, atomics, and bonds.
        """
        # Unpack prior
        coords = data["coords"]  # (B, N_l, C)
        atomics = data["atomics"]  # (B, N_l, d)
        bonds = data["bonds"]  # (B, N_l, N_l, n_bonds)
        interactions = data["interactions"]  # (B, N_l, N_p, n_interactions)

        # Unpack predicted
        pred_coords = prediction["coords"]  # (B, N_l, C)
        pred_atomics = prediction["atomics"]  # (B, N_l, d)
        pred_bonds = prediction["bonds"]  # (B, N_l, N_l, n_bonds)
        lig_mask = data["mask"].bool()  # (B, N_l)

        # Interaction mask
        inpaint_mask = interactions[..., 1:].sum(dim=(2, 3)) > 0
        inpaint_mask = inpaint_mask & lig_mask

        # Overwrite coordinates and atomics where interactions are present
        pred_coords[inpaint_mask] = coords[inpaint_mask]
        pred_atomics[inpaint_mask] = atomics[inpaint_mask]

        # Overwrite bonds with a pairwise fixed mask:
        inpaint_mask = inpaint_mask.unsqueeze(2) & inpaint_mask.unsqueeze(1)
        bond_mask = torch.argmax(bonds, dim=-1) != 0
        fixed_mask = inpaint_mask & bond_mask
        pred_bonds[fixed_mask] = bonds[fixed_mask]
        # pred_bonds = torch.where(fixed_mask, bonds, pred_bonds)

        out = {
            "coords": pred_coords,
            "atomics": pred_atomics,
            "bonds": pred_bonds,
            "mask": data["mask"],
        }
        if "charges" in prediction:
            out["charges"] = prediction["charges"]
        if keep_interactions and "interactions" in prediction:
            out["interactions"] = prediction["interactions"]

        return out

    def undo_zero_com(self, coords, com):
        return coords + com

    def undo_zero_com_batch(self, coords, node_mask, com_list):
        shifted_coords = torch.zeros_like(coords)
        for i in range(coords.size(0)):
            shifted = self.undo_zero_com(coords[i], com_list[i]) * node_mask[
                i
            ].unsqueeze(-1)
            shifted_coords[i] = shifted
        return shifted_coords


# *********************************************************************************************************************
# ******************************************** Lightning Flow Matching Models *****************************************
# *********************************************************************************************************************


class LigandPocketCFM(pl.LightningModule):
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
        interaction_loss_weight: float = 1.0,
        charge_loss_weight: float = 1.0,
        pocket_noise: str = "random",
        ligand_only: bool = False,
        pairwise_metrics: bool = True,
        use_ema: bool = True,
        compile_model: bool = True,
        self_condition: bool = False,
        remove_hs: bool = False,
        distill: bool = False,
        lr_schedule: str = "constant",
        lr_gamma: float = 0.998,
        sampling_strategy: str = "linear",
        warm_up_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        train_smiles: Optional[list[str]] = None,
        type_mask_index: Optional[int] = None,
        bond_mask_index: Optional[int] = None,
        flow_interactions: bool = False,
        predict_interactions: bool = False,
        interaction_inpainting: bool = False,
        func_group_inpainting: bool = False,
        scaffold_inpainting: bool = False,
        fragment_inpainting: bool = False,
        linker_inpainting: bool = False,
        substructure_inpainting: bool = False,
        corrector_iters: int = None,
        use_t_loss_weights: bool = False,
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
        self.interaction_loss_weight = interaction_loss_weight
        self.charge_loss_weight = charge_loss_weight
        self.pairwise_metrics = pairwise_metrics
        self.compile_model = compile_model
        self.self_condition = self_condition
        self.distill = distill
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.sampling_strategy = sampling_strategy
        self.warm_up_steps = warm_up_steps
        self.total_steps = total_steps
        self.type_mask_index = type_mask_index
        self.bond_mask_index = bond_mask_index
        self.pocket_noise = pocket_noise
        self.ligand_only = ligand_only
        self.flow_interactions = flow_interactions
        self.predict_interactions = predict_interactions
        self.interaction_inpainting = interaction_inpainting
        self.func_group_inpainting = func_group_inpainting
        self.scaffold_inpainting = scaffold_inpainting
        self.fragment_inpainting = fragment_inpainting
        self.linker_inpainting = linker_inpainting
        self.substructure_inpainting = substructure_inpainting
        self.corrector_iters = corrector_iters
        self.use_t_loss_weights = use_t_loss_weights

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
            "flow_interactions": flow_interactions,
            "predict_interactions": predict_interactions,
            "interaction_inpainting": interaction_inpainting,
            "corrector_iters": corrector_iters,
            "remove_hs": remove_hs,
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
        self.gen_dist_metrics = None
        if self.dataset_info is not None:
            gen_dist_metrics = Metrics.DistributionDistance(
                dataset_info=self.dataset_info
            )
            self.gen_dist_metrics = MetricCollection(
                {"distribution-distance": gen_dist_metrics}, compute_groups=False
            )

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

    def forward(
        self,
        batch,
        pocket_batch,
        t,
        training=False,
        cond_batch=None,
        pocket_equis=None,
        pocket_invs=None,
    ):
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
        mask = batch["mask"]

        pocket_coords = pocket_batch["coords"]
        pocket_atoms = pocket_batch["atom_names"]
        pocket_bonds = pocket_batch["bonds"]
        pocket_charges = pocket_batch["charges"]
        pocket_res = pocket_batch["res_names"]
        pocket_mask = pocket_batch["mask"]

        interactions = batch["interactions"] if self.flow_interactions else None

        # Prepare invariant atom features
        ligand_times_cont = t[0].view(-1, 1, 1).expand(-1, coords.size(1), -1)
        ligand_times_disc = t[1].view(-1, 1, 1).expand(-1, coords.size(1), -1)
        # pocket_times = t[2].view(-1, 1, 1).expand(-1, pocket_coords.size(1), -1) # rigid pocket, not needed
        interaction_times = (
            t[-1].view(-1, 1, 1).expand(-1, coords.size(1), -1)
            if self.flow_interactions or self.predict_interactions
            else None
        )
        if (
            self.interaction_inpainting
            or self.func_group_inpainting
            or self.scaffold_inpainting
            or self.linker_inpainting
            or self.substructure_inpainting
            or self.fragment_inpainting
        ):
            inpaint_mask = batch["fragment_mask"].bool()
            ligand_times_cont = self.builder._inpaint_times(
                ligand_times_cont.squeeze(), inpaint_mask
            ).unsqueeze(-1)
            ligand_times_disc = self.builder._inpaint_times(
                ligand_times_disc.squeeze(), inpaint_mask
            ).unsqueeze(-1)
        times = [ligand_times_cont, ligand_times_disc, interaction_times]

        if cond_batch is not None:
            out = self.gen(
                coords,
                torch.argmax(atom_types, dim=-1),
                torch.argmax(bonds, dim=-1),
                atom_mask=mask,
                extra_feats=times,
                cond_coords=cond_batch["coords"],
                cond_atomics=cond_batch["atomics"],
                cond_bonds=cond_batch["bonds"],
                pocket_coords=pocket_coords,
                pocket_atom_names=pocket_atoms,
                pocket_atom_charges=torch.argmax(pocket_charges, dim=-1),
                pocket_bond_types=torch.argmax(pocket_bonds, dim=-1),
                pocket_res_types=pocket_res,
                pocket_atom_mask=pocket_mask,
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
                interactions=(
                    torch.argmax(interactions, dim=-1)
                    if interactions is not None
                    else None
                ),
            )
        else:
            out = self.gen(
                coords,
                torch.argmax(atom_types, dim=-1),
                torch.argmax(bonds, dim=-1),
                atom_mask=mask,
                extra_feats=times,
                pocket_coords=coords,
                pocket_atom_names=pocket_atoms,
                pocket_atom_charges=pocket_charges,
                pocket_res_types=pocket_res,
                pocket_bond_types=torch.argmax(pocket_bonds, dim=-1),
                pocket_atom_mask=pocket_mask,
                pocket_equis=pocket_equis,
                pocket_invs=pocket_invs,
                interactions=(
                    torch.argmax(interactions, dim=-1)
                    if interactions is not None
                    else None
                ),
            )

        return out

    def training_step(self, batch, b_idx):
        # Input data
        _, data, interpolated, times = batch

        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = data["interactions"]
        # Extract ligand data
        lig_interp = self.builder.extract_ligand_from_complex(interpolated)
        lig_interp["fragment_mask"] = data["fragment_mask"]
        lig_interp["interactions"] = interpolated["interactions"]
        lig_data = self.builder.extract_ligand_from_complex(data)
        lig_data["interactions"] = data["interactions"]
        lig_data["pocket_mask"] = pocket_data["mask"]
        times = [times[:, 0], times[:, 1], times[:, 2], times[:, 3]]

        cond_batch = None

        # If training with self conditioning, half the time generate a conditional batch by setting cond to zeros
        if self.self_condition:

            cond_batch = {
                "coords": torch.zeros_like(lig_interp["coords"]),
                "atomics": torch.zeros_like(lig_interp["atomics"]),
                "bonds": torch.zeros_like(lig_interp["bonds"]),
            }

            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    cond_out = self(
                        lig_interp,
                        pocket_data,
                        times,
                        training=True,
                        cond_batch=cond_batch,
                    )
                    cond_batch = {
                        "coords": cond_out[0],
                        "atomics": F.softmax(cond_out[1], dim=-1),
                        "bonds": F.softmax(cond_out[2], dim=-1),
                    }

        out = self(lig_interp, pocket_data, times, training=True, cond_batch=cond_batch)
        predicted = {
            "coords": out[0],
            "atomics": out[1],
            "bonds": out[2],
            "charges": out[3],
            "mask": lig_data["mask"],
        }
        if self.predict_interactions or self.flow_interactions:
            predicted["interactions"] = out[4]

        # import pdb

        # pdb.set_trace()
        # lig_prior = self.builder.extract_ligand_from_complex(batch[0])
        # self.builder.tensors_to_xyz(
        #     prior=lig_prior,
        #     interpolated=lig_interp,
        #     data=lig_data,
        #     coord_scale=self.hparams.coord_scale,
        #     idx=0,
        #     save_dir=f"{self.hparams.save_dir}/tmp",
        # )

        # Get the ligand times if timestep-dependent loss weights are used
        ligand_times = None
        if self.use_t_loss_weights:
            inpaint_mask = lig_interp["fragment_mask"].bool()
            ligand_times = self.builder._inpaint_times(times[0], inpaint_mask)

        losses = self._loss(lig_data, lig_interp, predicted, times=ligand_times)
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

        return loss

    def validation_step(self, batch, b_idx):
        # Input data
        prior, data, interpolated, interp_times = batch

        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = prior["interactions"]
        # Extract ligand data
        lig_prior = self.builder.extract_ligand_from_complex(prior)
        lig_prior["fragment_mask"] = prior["fragment_mask"]
        lig_prior["interactions"] = prior["interactions"]

        # Build starting times for the integrator
        lig_times_cont = torch.zeros(prior["coords"].size(0), device=self.device)
        lig_times_disc = torch.zeros(prior["coords"].size(0), device=self.device)
        pocket_times = torch.zeros(pocket_data["coords"].size(0), device=self.device)
        interaction_times = torch.zeros(prior["coords"].size(0), device=self.device)
        prior_times = [lig_times_cont, lig_times_disc, pocket_times, interaction_times]

        # Generate
        gen_batch = self._generate(
            lig_prior,
            pocket_data,
            steps=self.integrator.steps,
            times=prior_times,
            strategy=self.sampling_strategy,
            corr_iters=self.corrector_iters,
        )
        gen_mols = self._generate_mols(gen_batch)

        if not self.trainer.sanity_checking:
            self.gen_mol_metrics.update(gen_mols)
            if self.gen_dist_metrics is not None:
                self.gen_dist_metrics.update(gen_mols)

            # Also measure the model's ability to recreate the original molecule when a bit of prior noise has been added
            if self.pairwise_metrics:
                gen_interp_steps = max(
                    1, int((1 - interp_times[0][0].item()) * self.integrator.steps)
                )
                gen_interp_batch = self._generate(
                    interpolated,
                    pocket_data,
                    times=interp_times,
                    steps=gen_interp_steps,
                )
                gen_interp_mols = self._generate_mols(gen_interp_batch)
                data_mols = self._generate_mols(data, scale=self.coord_scale)
                self.pair_metrics.update(gen_interp_mols, data_mols)

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            gen_dist_results, pair_metrics_results = {}, {}

            gen_metrics_results = self.gen_mol_metrics.compute()
            if self.gen_dist_metrics is not None:
                gen_dist_results = self.gen_dist_metrics.compute()

            metrics = {
                **gen_metrics_results,
                **pair_metrics_results,
                **gen_dist_results,
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
            if self.gen_dist_metrics is not None:
                self.gen_dist_metrics.reset()

            if self.pairwise_metrics:
                self.pair_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def predict_step(self, batch, batch_idx):
        # input data
        prior, data, interpolated, _ = batch

        # Extract pocket data
        pocket_data = self.builder.extract_pocket_from_complex(data)
        pocket_data["interactions"] = data["interactions"]
        pocket_data["complex"] = data["complex"]
        # Extract ligand data
        lig_prior = self.builder.extract_ligand_from_complex(prior)
        lig_prior["fragment_mask"] = prior["fragment_mask"]
        lig_prior["interactions"] = prior["interactions"]

        # specify times
        lig_times_cont = torch.zeros(prior["coords"].size(0), device="cuda")
        lig_times_disc = torch.zeros(prior["coords"].size(0), device="cuda")
        pocket_times = torch.zeros(pocket_data["coords"].size(0), device="cuda")
        interaction_times = torch.zeros(prior["coords"].size(0), device="cuda")
        prior_times = [lig_times_cont, lig_times_disc, pocket_times, interaction_times]

        # Generate
        output = self._generate(
            lig_prior,
            pocket_data,
            steps=self.integrator.steps,
            times=prior_times,
            strategy=self.sampling_strategy,
            iter=batch_idx,
            save_traj=False,
            corr_iters=self.corrector_iters,
        )
        gen_ligs = self._generate_mols(output)

        # retrieve ground truth/native ligands and pdbs
        ref_ligs = self._generate_ligs(
            data, lig_mask=data["lig_mask"].bool(), scale=self.coord_scale
        )
        ref_ligs_with_hs = self.retrieve_ligs_with_hs(data)
        ref_pdbs = self.retrieve_pdbs(
            data, save_dir=Path(self.hparams.save_dir) / "ref_pdbs"
        )
        ref_pdbs_with_hs = self.retrieve_pdbs_with_hs(
            data, save_dir=Path(self.hparams.save_dir) / "ref_pdbs"
        )

        # group ligands by pdb in a list of lists as we are potentially sampling N ligands per target;
        # de-duplicate native ligands and pdbs as they are loaded N times
        ligs_by_pdb = defaultdict(create_list_defaultdict)
        ligs_by_pdb_with_hs = defaultdict(create_list_defaultdict)
        for gen_lig, ref_lig, ref_lig_with_hs, pdb, pdb_with_hs in zip(
            gen_ligs, ref_ligs, ref_ligs_with_hs, ref_pdbs, ref_pdbs_with_hs
        ):
            ligs_by_pdb[pdb]["gen"].append(gen_lig)
            ligs_by_pdb[pdb]["ref"] = ref_lig
            ligs_by_pdb_with_hs[pdb_with_hs]["ref"] = ref_lig_with_hs
        gen_ligs_by_pdb = [v["gen"] for _, v in ligs_by_pdb.items()]
        ref_ligs = [v["ref"] for v in ligs_by_pdb.values()]
        ref_ligs_with_hs = [v["ref"] for v in ligs_by_pdb_with_hs.values()]
        ref_pdbs = [pdb for pdb in ligs_by_pdb]
        ref_pdbs_with_hs = [pdb for pdb in ligs_by_pdb_with_hs]

        outputs = {
            "gen_ligs": gen_ligs_by_pdb,
            "native_ligs": ref_ligs,
            "native_ligs_with_hs": ref_ligs_with_hs,
            "ref_pdbs": ref_pdbs,
            "ref_pdbs_with_hs": ref_pdbs_with_hs,
        }
        return outputs

    # def on_predict_epoch_end(self):
    #     gen_metrics_results = self.gen_mol_metrics.compute()
    #     gen_dist_results = self.gen_dist_metrics.compute()
    #     posebusters_validity = self.posebusters_validity.compute()
    #     genbench3d_validity = self.genbench3d_validity.compute()
    #     genbench3d_sb_validity = self.genbench3d_sb_validity.compute()
    #     interaction_recovery = self.interaction_recovery.compute()

    #     metrics = {
    #         **gen_metrics_results,
    #         **gen_dist_results,
    #         **posebusters_validity,
    #         **genbench3d_validity,
    #         **genbench3d_sb_validity,
    #         **interaction_recovery,
    #     }

    #     torch.save(metrics, os.path.join(self.hparams.save_dir, "metrics.pt"))
    #     if self.local_rank == 0:
    #         print_results(metrics)

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
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=self.lr_gamma)
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

    def _loss(self, data, interpolated, predicted, times=None):
        pred_coords = predicted["coords"]

        coords = data["coords"]
        mask = data["mask"]
        if self.use_t_loss_weights:
            assert times is not None, "t_loss_weights requires times to be passed in"
            t_loss_weights = times / (1 - times)
            t_loss_weights = torch.clamp(t_loss_weights, min=0.05, max=1.5).squeeze(1)

        coord_loss = F.mse_loss(pred_coords, coords, reduction="none")
        n_atoms = mask.unsqueeze(-1).sum(dim=(1, 2))
        coord_loss = (coord_loss * mask.unsqueeze(-1)).sum(dim=(1, 2)) / n_atoms
        coord_loss = coord_loss.mean()
        type_loss = self._type_loss(data, interpolated, predicted)
        bond_loss = self._bond_loss(data, interpolated, predicted)
        charge_loss = self._charge_loss(data, predicted)
        if self.predict_interactions or self.flow_interactions:
            interaction_loss = self._interaction_loss(data, interpolated, predicted)

        losses = {
            "coord-loss": coord_loss,
            "type-loss": type_loss,
            "bond-loss": bond_loss,
            "charge-loss": charge_loss,
        }
        if self.predict_interactions or self.flow_interactions:
            losses["interaction-loss"] = interaction_loss
        return losses

    def _type_loss(self, data, interpolated, predicted, t_loss_weights=None, eps=1e-3):

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

        type_loss = (type_loss * mask).sum(dim=(1, 2)) / n_atoms
        if t_loss_weights:
            type_loss = type_loss * t_loss_weights
        type_loss = type_loss.mean() * self.type_loss_weight
        return type_loss

    def _bond_loss(self, data, interpolated, predicted, t_loss_weights=None, eps=1e-3):
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

        bond_loss = (bond_loss * adj_matrix).sum(dim=(1, 2)) / n_bonds
        if t_loss_weights:
            bond_loss = bond_loss * t_loss_weights
        bond_loss = bond_loss.mean() * self.bond_loss_weight
        return bond_loss

    def _interaction_loss(self, data, interpolated, predicted, eps=1e-3):
        gamma = 2.0
        alpha = (
            torch.tensor([0.5, 1.6, 1.6, 1.6, 1.7, 1.8, 1.8, 1.4, 2.0, 2.0])
            .float()
            .to(self.device)
        )
        lig_mask = data["mask"].bool()
        pocket_mask = data["pocket_mask"].bool()
        combined_mask = pocket_mask[:, :, None] * lig_mask[:, None, :]

        pred_logits = predicted["interactions"].permute(0, 2, 1, 3)
        interactions = torch.argmax(data["interactions"].permute(0, 2, 1, 3), dim=-1)

        num_actual_interactions = max(interactions.count_nonzero(dim=-1).sum(), 1)

        ce_loss = F.cross_entropy(
            pred_logits.flatten(0, 2),
            interactions.flatten(0, 2),
            reduction="none",
        )
        ce_loss = ce_loss.unflatten(
            0, (pred_logits.size(0), pred_logits.size(1), pred_logits.size(2))
        )
        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** gamma

        if isinstance(alpha, torch.Tensor):
            alpha_factor = alpha[interactions]
        else:
            alpha_factor = alpha

        interaction_loss = focal_factor * alpha_factor * ce_loss
        interaction_loss = (interaction_loss * combined_mask).sum(
            dim=(1, 2)
        ) / num_actual_interactions  # (
        # combined_mask.sum(dim=(1, 2)) + eps
        # )
        return interaction_loss.mean() * self.interaction_loss_weight

    def _charge_loss(self, data, predicted, t_loss_weights=None, eps=1e-3):
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

        charge_loss = (charge_loss * mask).sum(dim=1) / n_atoms
        if t_loss_weights:
            charge_loss = charge_loss * t_loss_weights
        charge_loss = charge_loss.mean() * self.charge_loss_weight
        return charge_loss

    def _build_trajectory(
        self,
        curr,
        predicted,
        pocket_data,
        iter: int = 0,
        step: int = 0,
    ):
        pred_complex = self.builder.add_ligand_to_pocket(
            lig_data=predicted,
            pocket_data=pocket_data,
        )
        self.builder.write_xyz_file_from_batch(
            data=pred_complex,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
            t=step,
        )
        # Move predictions to CPU
        predictions = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted.items()
        }
        # Undo zero COM alignment if necessary
        if "complex" in pocket_data:
            predictions["coords"] = self.builder.undo_zero_com_batch(
                predictions["coords"],
                predictions["mask"],
                com_list=[system.com for system in pocket_data["complex"]],
            )
        self.builder.write_xyz_file_from_batch(
            data=predictions,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            t=step,
        )
        # Move current state to CPU
        curr_ = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in curr.items()
        }
        # Undo zero COM alignment if necessary
        if "complex" in pocket_data:
            curr_["coords"] = self.builder.undo_zero_com_batch(
                curr_["coords"],
                curr_["mask"],
                com_list=[system.com for system in pocket_data["complex"]],
            )
        self.builder.write_xyz_file_from_batch(
            data=curr_,
            coord_scale=self.coord_scale,
            path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
            t=step,
        )

    def _save_trajectory(
        self,
        predicted,
        iter: int = 0,
    ):
        pred_mols = self._generate_mols(predicted)
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_pred_mols_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_pred_mols", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_pred_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_pred", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )
        self.builder.write_trajectory_as_xyz(
            pred_mols=pred_mols,
            file_path=os.path.join(self.hparams.save_dir, f"traj_interp_{iter}"),
            save_path=os.path.join(
                self.hparams.save_dir, "trajectories_interp", f"traj_{iter}"
            ),
            remove_intermediate_files=True,
        )

    def _generate(
        self,
        prior: dict,
        pocket_data: dict,
        steps: int,
        times: list,
        strategy: str = "linear",
        save_traj: bool = False,
        iter: int = 0,
        corr_iters=None,
        corr_step_size=None,
    ):

        corr_iters = 0 if corr_iters is None else corr_iters

        if strategy == "linear":
            time_points = np.linspace(0, 0.999, steps + 1).tolist()

        elif strategy == "log":
            time_points = (1 - np.geomspace(0.01, 0.999, steps + 1)).tolist()
            time_points.reverse()
            # time_points = 1.0 - torch.logspace(-2, 0, steps + 1).flip(0)
            # time_points = time_points - torch.min(time_points)
            # time_points = time_points / torch.max(time_points)
        else:
            raise ValueError(f"Unknown ODE integration strategy '{strategy}'")

        curr = {
            k: (
                v.clone()
                if torch.is_tensor(v)
                else v.copy() if isinstance(v, list) else v
            )
            for k, v in prior.items()
        }
        cond_batch = {
            "coords": torch.zeros_like(prior["coords"]),
            "atomics": torch.zeros_like(prior["atomics"]),
            "bonds": torch.zeros_like(prior["bonds"]),
        }

        step_sizes = [t1 - t0 for t0, t1 in zip(time_points[:-1], time_points[1:])]
        with torch.no_grad():
            # Generate pocket encodings only once at inference (NOTE: only for rigid SBDD)
            pocket_equis, pocket_invs = self.gen.get_pocket_encoding(
                pocket_data["coords"],
                pocket_data["atom_names"],
                pocket_atom_charges=torch.argmax(pocket_data["charges"], dim=-1),
                pocket_bond_types=torch.argmax(pocket_data["bonds"], dim=-1),
                pocket_res_types=pocket_data["res_names"],
                pocket_atom_mask=pocket_data["mask"],
            )
            for i, step_size in enumerate(step_sizes):
                cond = cond_batch if self.self_condition else None
                out = self(
                    curr,
                    pocket_data,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                )
                coords = out[0]
                type_probs = F.softmax(out[1], dim=-1)
                bond_probs = F.softmax(out[2], dim=-1)
                charge_probs = F.softmax(out[3], dim=-1)
                mask = out[-1]
                if self.predict_interactions or self.flow_interactions:
                    interaction_probs = F.softmax(out[4], dim=-1)

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
                if self.flow_interactions:
                    predicted["interactions"] = interaction_probs
                curr = self.integrator.step(curr, predicted, prior, times, step_size)

                if (
                    self.interaction_inpainting
                    or self.func_group_inpainting
                    or self.scaffold_inpainting
                    or self.linker_inpainting
                    or self.substructure_inpainting
                    or self.fragment_inpainting
                ):
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )
                # Save trajectory
                if save_traj:
                    self._build_trajectory(
                        curr,
                        predicted,
                        pocket_data,
                        iter=iter,
                        step=i,
                    )

                lig_times_cont = times[0] + step_size
                lig_times_disc = times[1] + step_size
                pocket_times = times[2] + step_size
                interaction_times = times[-1] + step_size
                times = [
                    lig_times_cont,
                    lig_times_disc,
                    pocket_times,
                    interaction_times,
                ]

            # Corrector iterations at the end of sampling
            for _ in range(corr_iters):
                cond = cond_batch if self.self_condition else None
                out = self(
                    curr,
                    pocket_data,
                    times,
                    training=False,
                    cond_batch=cond,
                    pocket_equis=pocket_equis,
                    pocket_invs=pocket_invs,
                )

                coords = out[0]
                type_probs = F.softmax(out[1], dim=-1)
                bond_probs = F.softmax(out[2], dim=-1)
                charge_probs = F.softmax(out[3], dim=-1)
                mask = out[-1]
                if self.predict_interactions or self.flow_interactions:
                    interaction_probs = F.softmax(out[4], dim=-1)

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
                    "mask": curr["mask"],
                }
                if self.flow_interactions:
                    predicted["interactions"] = interaction_probs

                step_size = 1 / steps if corr_step_size is None else corr_step_size
                curr = self.integrator.corrector_iter(
                    curr, predicted, prior, times, step_size
                )
                if (
                    self.interaction_inpainting
                    or self.func_group_inpainting
                    or self.scaffold_inpainting
                    or self.linker_inpainting
                    or self.substructure_inpainting
                    or self.fragment_inpainting
                ):
                    curr = self.builder.inpaint_molecule(
                        data=prior,
                        prediction=curr,
                        pocket_mask=pocket_data["mask"].bool(),
                        keep_interactions=self.flow_interactions,
                    )

        # Move everything to CPU
        predicted = {
            k: v.cpu().detach() if torch.is_tensor(v) else v
            for k, v in predicted.items()
        }
        # Scale back coordinates if necessary
        predicted["coords"] = predicted["coords"] * self.coord_scale
        # Undo zero COM alignment if necessary
        if "complex" in pocket_data:
            predicted["coords"] = self.builder.undo_zero_com_batch(
                predicted["coords"],
                predicted["mask"],
                com_list=[system.com for system in pocket_data["complex"]],
            )

        if save_traj:
            self._save_trajectory(
                predicted,
                iter=iter,
            )

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

    def _generate_ligs(self, generated, lig_mask, scale=1.0, sanitise=False):
        """
        Generate ligand mols from output tensors
        """
        coords = generated["coords"] * scale
        coords = self.builder.undo_zero_com_batch(
            coords, generated["mask"], [system.com for system in generated["complex"]]
        )
        atom_dists = generated["atomics"]
        bond_dists = generated["bonds"]
        charge_dists = generated["charges"]

        mols = self.builder.ligs_from_complex(
            coords,
            mask=lig_mask,
            atom_dists=atom_dists,
            bond_dists=bond_dists,
            charge_dists=charge_dists,
            sanitise=sanitise,
        )
        return mols

    def retrieve_ligs_with_hs(self, data, save_idx=None):
        """
        Retrieve native ligand mols with hydrogens from the PocketComplex data
        NOTE: depending on the data, the ligand may come without hydrogens (e.g., CrossDocked2020)
        """
        systems = data["complex"] if save_idx is None else [data["complex"][save_idx]]
        ligs = [system.ligand.orig_mol.to_rdkit() for system in systems]
        if save_idx is not None:
            return ligs[0]
        return ligs

    def _retrieve_pdbs(self, data, coords=None, iter="", stage="ref_val"):
        """
        Generate PDB files from pocket data.
        If (predicted) coords are provided, the PDB files will be generated using these coordinates,
        otherwise the (CoM aligned and potentially normalized) coordinates from the data will be used.
        """
        systems = data["complex"]
        pdb_path = Path(self.hparams.data_path) / f"{stage}_pdbs"
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + f"{iter}.pdb"))
            for system in systems
        ]
        if coords is None:
            mask = data["mask"]
            pocket_mask = data["pocket_mask"]
            # When no (predicted) coords are provided, use the coords from the data and undo zero COM alignment
            # NOTE: Predicted coordinates are always scaled back to the original coordinates and are not zero COM aligned anymore
            coords = data["coords"] * self.coord_scale
            coords = self.builder.undo_zero_com_batch(
                coords, mask, [system.com for system in systems]
            )
        else:
            pocket_mask = data["mask"]
        _ = [
            system.holo.set_coords(coord[mask.bool()].cpu().numpy()).write_pdb(
                pdb_file, include_bonds=True
            )
            for system, coord, mask, pdb_file in zip(
                systems, coords, pocket_mask, pdb_files
            )
        ]
        return pdb_files

    def retrieve_pdbs(self, data, save_dir, save_idx=None, coords=None, iter=""):
        """
        Retrieve PDB files from the PocketComplex data.
        """
        systems = (
            data["complex"] if save_idx is not None else [data["complex"][save_idx]]
        )
        pdb_path = Path(save_dir)
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + ".pdb"))
            for system in systems
        ]
        if coords is None:
            pdb_files = [
                str(pdb_path / (system.metadata["system_id"] + ".pdb"))
                for system in systems
            ]
            mask = data["mask"]
            pocket_mask = data["pocket_mask"]
            # When no (predicted) coords are provided, use the coords from the data and undo zero COM alignment
            # NOTE: Predicted coordinates are always scaled back to the original coordinates and are not zero COM aligned anymore
            coords = data["coords"] * self.coord_scale
            coords = self.builder.undo_zero_com_batch(
                coords, mask, [system.com for system in systems]
            )
        else:
            pdb_files = [
                str(pdb_path / (system.metadata["system_id"] + f"_{iter}{i}.pdb"))
                for i, system in enumerate(systems)
            ]
            pocket_mask = data["mask"]
        _ = [
            system.holo.set_coords(coord[mask.bool()].cpu().numpy()).write_pdb(
                pdb_file, include_bonds=True
            )
            for system, coord, mask, pdb_file in zip(
                systems, coords, pocket_mask, pdb_files
            )
        ]
        if save_idx is not None:
            return pdb_files[0]
        return pdb_files

    def _retrieve_pdbs_with_hs(self, data, stage="ref_val", pocket_type="holo"):
        """
        Generate PDB files with hydrogens from the PocketComplex data considering CoM alignment
        """
        systems = data["complex"]
        pdb_path = Path(self.hparams.data_path) / f"{stage}_pdbs"
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + "_with_hs.pdb"))
            for system in systems
        ]
        if pocket_type == "apo":
            for system, pdb_file in zip(systems, pdb_files):
                system.apo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        elif pocket_type == "holo":
            for system, pdb_file in zip(systems, pdb_files):
                system.holo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        else:
            raise ValueError(f"Unknown pocket type '{pocket_type}'")

        return pdb_files

    def retrieve_pdbs_with_hs(self, data, save_dir, save_idx=None, pocket_type="holo"):
        """
        Generate PDB files with hydrogens from the PocketComplex data considering CoM alignment
        """
        systems = (
            data["complex"] if save_idx is not None else [data["complex"][save_idx]]
        )
        pdb_path = Path(save_dir)
        pdb_path.mkdir(parents=True, exist_ok=True)
        pdb_files = [
            str(pdb_path / (system.metadata["system_id"] + "_with_hs.pdb"))
            for system in systems
        ]
        if pocket_type == "apo":
            for system, pdb_file in zip(systems, pdb_files):
                system.apo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        elif pocket_type == "holo":
            for system, pdb_file in zip(systems, pdb_files):
                system.holo.orig_pocket.write_pdb(pdb_file, include_bonds=True)
        else:
            raise ValueError(f"Unknown pocket type '{pocket_type}'")

        if save_idx is not None:
            return pdb_files[0]
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
