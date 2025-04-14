import logging
import math
import os
import pickle
import random
from collections import Counter
from typing import Optional

import numpy as np
import rdkit
import torch
import torch.distributed as td
import torch.nn.functional as F
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
from torch.utils.data import (
    BatchSampler,
    Dataset,
    DistributedSampler,
    RandomSampler,
    Sampler,
)

RDLogger.DisableLog("rdApp.*")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


_logger = logging.getLogger(__name__)


def get_ddp_info():
    """Get the global rank and the total number of processes."""
    try:
        rank = td.get_rank()
        world_size = td.get_world_size()
    except (ValueError, RuntimeError):
        rank = 0
        world_size = 1
    _logger.debug(f"DDP info: {rank}, {world_size}")
    return rank, world_size


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        sampler,
        bucket_limits,
        lengths,
        batch_cost,
        bucket_costs=None,
        drop_last=False,
        round_batch_to_8=False,
        seed=None,
        shuffle=True,
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit training)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError(
                "Largest length cannot be larger than largest bucket limit."
            )

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        bucket_costs = (
            [1] * len(bucket_limits) if bucket_costs is None else bucket_costs
        )
        bucket_batch_sizes = [
            self._round_batch_size(batch_cost / cost) for cost in bucket_costs
        ]

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for seq_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(seq_idx)
                    break

        if isinstance(sampler, DistributedSampler):
            self.rank = sampler.rank
            self.num_replicas = sampler.num_replicas

            # Create a batch sampler for each bucket
            samplers = []
            for b_idx, (idxs, batch_size) in enumerate(
                zip(buckets, bucket_batch_sizes)
            ):
                if len(idxs) == 0:
                    samplers.append(None)
                    continue

                sampler = DistributedSampler(
                    idxs,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    shuffle=shuffle,
                    seed=(self.seed + b_idx),
                    drop_last=drop_last,
                )
                batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
                samplers.append(batch_sampler)

        else:
            samplers = []
            for b_idx, (idxs, batch_size) in enumerate(
                zip(buckets, bucket_batch_sizes)
            ):
                if len(idxs) == 0:
                    samplers.append(None)
                    continue

                sampler = RandomSampler(idxs, replacement=False)
                batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
                samplers.append(batch_sampler)

        batches_per_bucket = [
            len(sampler) if sampler is not None else 0 for sampler in samplers
        ]

        print(f"\nitems per bucket", [len(idxs) for idxs in buckets])
        print("bucket batch sizes", bucket_batch_sizes)
        print("batches per bucket", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [
            iter(sampler) if sampler is not None else None for sampler in self.samplers
        ]
        remaining_batches = self.batches_per_bucket[:]

        while sum(remaining_batches) > 0:
            weights = np.array(remaining_batches) / sum(remaining_batches)
            b_idx = self.batch_idx_generator.choice(len(remaining_batches), p=weights)
            indices_in_bucket = next(iters[b_idx])
            batch = [self.buckets[b_idx][idx] for idx in indices_in_bucket]
            remaining_batches[b_idx] -= 1
            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs

    def set_epoch(self, epoch: int) -> None:
        """This needs to be called to ensure the distributed samplers have different behavious across epochs"""

        for batch_sampler in self.samplers:
            batch_sampler.sampler.set_epoch(epoch)


class DistributedBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        bucket_limits: list[int],
        lengths: list[int],
        batch_cost: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        bucket_costs: Optional[list[float]] = None,
        shuffle: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        round_batch_to_8: Optional[bool] = False,
        seed: Optional[int] = None,
    ):

        # Modern GPUs can be more efficient when data is provided as a multiple of 8 (for 16-bit {split}ing)
        self.round_batch_to_8 = round_batch_to_8
        self.drop_last = drop_last
        self.seed = torch.initial_seed() if seed is None else seed

        if bucket_costs is not None and len(bucket_costs) != len(bucket_limits):
            raise ValueError("The number of costs and buckets must be the same.")

        if max(lengths) > max(bucket_limits):
            raise ValueError(
                "Largest length cannot be larger than largest bucket limit."
            )

        bucket_limits = sorted(bucket_limits)

        # Use a constant bucket cost by default
        bucket_costs = (
            [1] * len(bucket_limits) if bucket_costs is None else bucket_costs
        )
        bucket_batch_sizes = [
            self._round_batch_size(batch_cost / cost) for cost in bucket_costs
        ]

        # Add indices to correct bucket based on seq length
        buckets = [[] for _ in range(len(bucket_limits))]
        for seq_idx, length in enumerate(lengths):
            for b_idx, limit in enumerate(bucket_limits):
                if limit >= length:
                    buckets[b_idx].append(seq_idx)
                    break

        # Create a batch sampler for each bucket
        samplers = []
        for b_idx, (idxs, batch_size) in enumerate(zip(buckets, bucket_batch_sizes)):
            if len(idxs) == 0:
                samplers.append(None)
                continue

            sampler = DistributedSampler(
                idxs,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=(self.seed + b_idx),
                drop_last=drop_last,
            )
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
            samplers.append(batch_sampler)

        batches_per_bucket = [
            len(sampler) if sampler is not None else 0 for sampler in samplers
        ]

        print()
        print("items per bucket", [len(idxs) for idxs in buckets])
        print("bucket batch sizes", bucket_batch_sizes)
        print("batches per bucket", batches_per_bucket)

        self.buckets = buckets
        self.samplers = samplers
        self.bucket_batch_sizes = bucket_batch_sizes
        self.batches_per_bucket = batches_per_bucket
        self.batch_idx_generator = np.random.default_rng(self.seed)

    def __len__(self):
        return sum(self.batches_per_bucket)

    def __iter__(self):
        iters = [
            iter(sampler) if sampler is not None else None for sampler in self.samplers
        ]
        remaining_batches = self.batches_per_bucket[:]

        while sum(remaining_batches) > 0:
            weights = np.array(remaining_batches) / sum(remaining_batches)
            b_idx = self.batch_idx_generator.choice(len(remaining_batches), p=weights)
            indices_in_bucket = next(iters[b_idx])
            batch = [self.buckets[b_idx][idx] for idx in indices_in_bucket]
            remaining_batches[b_idx] -= 1
            yield batch

    def _round_batch_size(self, batch_size):
        if not self.round_batch_to_8:
            bs = math.floor(batch_size)
        else:
            bs = 8 * round(batch_size / 8)

        bs = 1 if bs == 0 else bs
        return bs

    def set_epoch(self, epoch: int) -> None:
        """This needs to be called to ensure the distributed samplers have different behavious across epochs"""

        for batch_sampler in self.samplers:
            batch_sampler.sampler.set_epoch(epoch)


class BatchSamplerSimilarLength(Sampler):
    def __init__(self, lengths, batch_size, indices=None, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # get the indicies and length
        self.indices = [(i, src_len) for i, src_len in enumerate(lengths)]
        # if indices are passed, then use only the ones passed (for ddp)
        if indices is not None:
            self.indices = torch.tensor(self.indices)[indices].tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * 100):
            pooled_indices.extend(
                sorted(self.indices[i : i + self.batch_size * 100], key=lambda x: x[1])
            )
        self.pooled_indices = [x[0] for x in pooled_indices]

        # yield indices for current batch
        batches = [
            self.pooled_indices[i : i + self.batch_size]
            for i in range(0, len(self.pooled_indices), self.batch_size)
        ]

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


class DistributedBatchSamplerSimilarLength(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=10,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = BatchSamplerSimilarLength(
            self.dataset.lengths, batch_size=self.batch_size, indices=indices
        )
        return iter(batch_sampler)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size


## UTILS ###


def mol_to_torch(
    mol,
    smiles=None,
    remove_hs: bool = False,
):
    import torch
    from rdkit import Chem

    if remove_hs:
        # mol = Chem.RemoveAllHs(mol)
        mol = Chem.RemoveHs(
            mol
        )  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    bond_indices = adj.nonzero().contiguous().T
    bond_indices = bond_indices[:, bond_indices[0] < bond_indices[1]]
    bond_types = adj[bond_indices[0], bond_indices[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hs:
        assert max(bond_types) < 4
    bond_types = bond_types.long()
    bond_indices = bond_indices.T

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    atom_types = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]).long()
    all_charges = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()]).long()

    return pos, atom_types, bond_indices, bond_types, all_charges, smiles


class Statistics:
    def __init__(
        self,
        num_nodes,
        atom_types,
        bond_types,
        charge_types,
        valencies,
        bond_lengths,
        bond_angles,
        dihedrals=None,
        is_in_ring=None,
        is_aromatic=None,
        hybridization=None,
        force_norms=None,
        is_h_donor=None,
        is_h_acceptor=None,
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.force_norms = force_norms
        self.is_h_donor = is_h_donor
        self.is_h_acceptor = is_h_acceptor

    @staticmethod
    def get_statistics(path, split, dataset, remove_hs=False):
        h = "noh" if remove_hs else "h"
        processed_paths = [
            f"{split}_{h}.pt",
            f"{split}_n_{h}.pickle",
            f"{split}_atom_types_{h}.npy",
            f"{split}_bond_types_{h}.npy",
            f"{split}_charges_{h}.npy",
            f"{split}_valency_{h}.pickle",
            f"{split}_bond_lengths_{h}.pickle",
            f"{split}_angles_{h}.npy",
            f"{split}_is_aromatic_{h}.npy",
            f"{split}_is_in_ring_{h}.npy",
            f"{split}_hybridization_{h}.npy",
        ]
        if dataset in ["plinder", "crossdocked", "kinodata", "kiba"]:
            processed_paths.extend(
                [
                    f"{split}_is_h_donor_{h}.npy",
                    f"{split}_is_h_acceptor_{h}.npy",
                    f"{split}_dihedrals_{h}.npy",
                ]
            )
        processed_paths = [f"{path}/{p}" for p in processed_paths]
        statistics = Statistics(
            num_nodes=load_pickle(processed_paths[1]),
            atom_types=torch.from_numpy(np.load(processed_paths[2])),
            bond_types=torch.from_numpy(np.load(processed_paths[3])),
            charge_types=torch.from_numpy(np.load(processed_paths[4])),
            valencies=load_pickle(processed_paths[5]),
            bond_lengths=load_pickle(processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(processed_paths[7])),
            is_aromatic=torch.from_numpy(np.load(processed_paths[8])).float(),
            is_in_ring=torch.from_numpy(np.load(processed_paths[9])).float(),
            hybridization=torch.from_numpy(np.load(processed_paths[10])).float(),
            is_h_donor=(
                torch.from_numpy(np.load(processed_paths[11])).float()
                if dataset in ["plinder", "crossdocked", "kinodata", "kiba"]
                else None
            ),
            is_h_acceptor=(
                torch.from_numpy(np.load(processed_paths[12])).float()
                if dataset in ["plinder", "crossdocked", "kinodata", "kiba"]
                else None
            ),
            dihedrals=(
                torch.from_numpy(np.load(processed_paths[13])).float()
                if dataset in ["plinder", "crossdocked", "kinodata", "kiba"]
                else None
            ),
        )

        return statistics


x_map = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    "is_h_donor": [False, True],
    "is_h_acceptor": [False, True],
}

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


def mol_to_dict(
    mol,
    atom_encoder,
    remove_hs: bool = False,
):

    is_aromatic = []
    if remove_hs:
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            for atom in mol.GetAtoms():
                is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass

    smiles = Chem.MolToSmiles(mol)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hs:
        assert max(bond_types) != 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    atom_types = []
    all_charges = []
    is_in_ring = []
    sp_hybridization = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(
            atom.GetFormalCharge()
        )  # TODO: check if implicit Hs should be kept
        if not remove_hs:
            is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(x_map["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(x_map["hybridization"].index(atom.GetHybridization()))

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    is_aromatic = torch.Tensor(is_aromatic).long()
    is_in_ring = torch.Tensor(is_in_ring).long()
    hybridization = torch.Tensor(sp_hybridization).long()
    feats = factory.GetFeaturesForMol(mol)
    donor_ids = []
    acceptor_ids = []
    for f in feats:
        if f.GetFamily().lower() == "donor":
            donor_ids.append(f.GetAtomIds())
        elif f.GetFamily().lower() == "acceptor":
            acceptor_ids.append(f.GetAtomIds())

    if len(donor_ids) > 0:
        donor_ids = np.concatenate(donor_ids)
    else:
        donor_ids = np.array([])

    if len(acceptor_ids) > 0:
        acceptor_ids = np.concatenate(acceptor_ids)
    else:
        acceptor_ids = np.array([])
    is_acceptor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
    is_donor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
    if len(donor_ids) > 0:
        is_donor[donor_ids] = 1
    if len(acceptor_ids) > 0:
        is_acceptor[acceptor_ids] = 1

    is_donor = torch.from_numpy(is_donor).long()
    is_acceptor = torch.from_numpy(is_acceptor).long()

    data = dotdict(
        {
            "x": atom_types,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "pos": pos,
            "charges": all_charges,
            "smiles": smiles,
            "is_in_ring": is_in_ring,
            "is_aromatic": is_aromatic,
            "is_h_donor": is_donor,
            "is_h_acceptor": is_acceptor,
            "hybridization": hybridization,
            "mol": mol,
            "num_nodes": mol.GetNumAtoms(),
        }
    )

    return data


def compute_all_statistics(
    data_list,
    atom_encoder,
    charges_dic,
    additional_feats: bool = True,
    include_force_norms: bool = False,
    normalize=True,
):
    num_nodes = node_counts(data_list)
    atom_types = atom_type_counts(
        data_list, num_classes=len(atom_encoder), normalize=normalize
    )
    print(f"Atom types: {atom_types}")
    bond_types = edge_counts(data_list, num_bond_types=5, normalize=normalize)
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(
        data_list,
        num_classes=len(atom_encoder),
        charges_dic=charges_dic,
        normalize=normalize,
    )
    print(f"Charge types: {charge_types}")
    valency = valency_count(data_list, atom_encoder, normalize=normalize)
    print("Valency: ", valency)

    bond_lengths = bond_lengths_counts(data_list, normalize=normalize)
    print("Bond lengths: ", bond_lengths)
    angles = bond_angles(data_list, atom_encoder, normalize=normalize)
    dihedrals = dihedral_angles(data_list, normalize=normalize)

    add_feats = {}
    add_feats.update(additional_feat_counts(data_list=data_list, normalize=normalize))

    return Statistics(
        num_nodes=num_nodes,
        atom_types=atom_types,
        bond_types=bond_types,
        charge_types=charge_types,
        valencies=valency,
        bond_lengths=bond_lengths,
        bond_angles=angles,
        dihedrals=dihedrals,
        **add_feats,
    )


def additional_feat_counts(
    data_list,
    keys: list = [
        "is_aromatic",
        "is_in_ring",
        "hybridization",
        "is_h_donor",
        "is_h_acceptor",
    ],
    normalize=True,
):
    print(f"Computing node counts for features = {str(keys)}")

    num_classes_list = [len(x_map.get(key)) for key in keys]
    counts_list = [np.zeros(num_classes) for num_classes in num_classes_list]

    for data in data_list:
        for i, key, num_classes in zip(range(len(keys)), keys, num_classes_list):
            x = torch.nn.functional.one_hot(data.get(key), num_classes=num_classes)
            counts_list[i] += x.sum(dim=0).numpy()

    if normalize:
        for i in range(len(counts_list)):
            counts_list[i] = counts_list[i] / counts_list[i].sum()
    print("Done")

    results = dict()
    for key, count in zip(keys, counts_list):
        results[key] = count

    print(results)

    return results


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def atom_type_counts(data_list, num_classes, normalize=True):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()

    if normalize:
        counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5, normalize=True):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = (
            torch.nn.functional.one_hot(
                data.edge_attr - 1, num_classes=num_bond_types - 1
            )
            .sum(dim=0)
            .numpy()
        )
        d[0] += num_non_edges
        d[1:] += edge_types
    if normalize:
        d = d / d.sum()
    return d


def charge_counts(data_list, num_classes, charges_dic, normalize=True):
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charges_dic)))

    for data in data_list:
        for atom, charge in zip(data.x, data.charges):
            assert charge in [-2, -1, 0, 1, 2, 3]
            d[atom.item(), charges_dic[charge.item()]] += 1

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    if normalize:
        d = d / s
    print("Done.")
    return d


def valency_count(data_list, atom_encoder, normalize=True):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_list:
        edge_attr = data.edge_attr
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    if normalize:
        # Normalizing the valency counts
        for atom_type in valencies.keys():
            s = sum(valencies[atom_type].values())
            for valency, count in valencies[atom_type].items():
                valencies[atom_type][valency] = count / s
    print("Done.")
    return valencies


def bond_lengths_counts(data_list, num_bond_types=5, normalize=True):
    """Compute the bond lenghts separetely for each bond type."""
    print("Computing bond lengths...")
    all_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for data in data_list:
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_bond_lenghts[bond_type][d.item()] += 1

    if normalize:
        # Normalizing the bond lenghts
        for bond_type in range(1, num_bond_types):
            s = sum(all_bond_lenghts[bond_type].values())
            for d, count in all_bond_lenghts[bond_type].items():
                all_bond_lenghts[bond_type][d] = count / s
    print("Done.")
    return all_bond_lenghts


def bond_angles(data_list, atom_encoder, normalize=True):
    print("Computing bond angles...")
    all_bond_angles = np.zeros((len(atom_encoder.keys()), 180 * 10 + 1))
    for data in data_list:
        assert not torch.isnan(data.pos).any()
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(
                        i, j, k
                    )
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(
                        torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6)
                    )
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)
                    all_bond_angles[data.x[i].item(), bin] += 1

    if normalize:
        # Normalizing the angles
        s = all_bond_angles.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        all_bond_angles = all_bond_angles / s
    print("Done.")
    return all_bond_angles


def dihedral_angles(data_list, normalize=True):
    def calculate_dihedral_angles(mol):
        def find_dihedrals(mol):
            torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
            torsionQuery = Chem.MolFromSmarts(torsionSmarts)
            matches = mol.GetSubstructMatches(torsionQuery)
            torsionList = []
            btype = []
            for match in matches:
                idx2 = match[0]
                idx3 = match[1]
                bond = mol.GetBondBetweenAtoms(idx2, idx3)
                jAtom = mol.GetAtomWithIdx(idx2)
                kAtom = mol.GetAtomWithIdx(idx3)
                if (
                    (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ) or (
                    (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ):
                    continue
                for b1 in jAtom.GetBonds():
                    if b1.GetIdx() == bond.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in kAtom.GetBonds():
                        if (b2.GetIdx() == bond.GetIdx()) or (
                            b2.GetIdx() == b1.GetIdx()
                        ):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        bt = bond.GetBondTypeAsDouble()
                        # bt = str(bond.GetBondType())
                        # if bond.IsInRing():
                        #     bt += '_R'
                        btype.append(bt)
                        torsionList.append((idx1, idx2, idx3, idx4))
            return np.asarray(torsionList), np.asarray(btype)

        dihedral_idx, dihedral_types = find_dihedrals(mol)

        coords = mol.GetConformer().GetPositions()
        t_angles = []
        for t in dihedral_idx:
            u1, u2, u3, u4 = coords[torch.tensor(t)]

            a1 = u2 - u1
            a2 = u3 - u2
            a3 = u4 - u3

            v1 = np.cross(a1, a2)
            v1 = v1 / (v1 * v1).sum(-1) ** 0.5
            v2 = np.cross(a2, a3)
            v2 = v2 / (v2 * v2).sum(-1) ** 0.5
            porm = np.sign((v1 * a3).sum(-1))
            rad = np.arccos(
                (v1 * v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1) + 1e-9) ** 0.5
            )
            if not porm == 0:
                rad = rad * porm
            t_angles.append(rad * 180 / torch.pi)

        return np.asarray(t_angles), dihedral_types

    generated_dihedrals = torch.zeros(5, 180 * 10 + 1)
    for d in data_list:
        mol = d.mol
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    if normalize:
        s = generated_dihedrals.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        generated_dihedrals = generated_dihedrals.float() / s

    return generated_dihedrals


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) is int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
    """preds and target are 1d tensors. They contain histograms for bins that are regularly spaced"""
    target = normalize(target) / step_size
    preds = normalize(preds) / step_size
    max_len = max(len(preds), len(target))
    preds = F.pad(preds, (0, max_len - len(preds)))
    target = F.pad(target, (0, max_len - len(target)))

    cs_target = torch.cumsum(target, dim=0)
    cs_preds = torch.cumsum(preds, dim=0)
    return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert (
        target.dim() == 1 and preds.shape == target.shape
    ), f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s
