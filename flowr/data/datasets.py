import itertools
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from flowr.util.molrepr import GeometricMolBatch
from flowr.util.pocket import PocketComplexBatch

# *** Util functions ***


def load_smol_data(data_path, smol_cls, remove_hs=False):
    data_path = Path(data_path)

    # TODO handle having a directory with batched data files
    if data_path.is_dir():
        raise NotImplementedError()

    # TODO maybe read in chunks if this is too big
    bytes_data = data_path.read_bytes()
    data = smol_cls.from_bytes(bytes_data, remove_hs=remove_hs)
    return data


def load_npz_data(data_path, smol_cls):
    data_path = Path(data_path)

    # TODO handle having a directory with batched data files
    if data_path.is_dir():
        raise NotImplementedError()

    data = smol_cls.from_numpy(data_path)
    return data


# *** Abstract class for all Smol data types ***


class SmolDataset(ABC, torch.utils.data.Dataset):
    def __init__(self, smol_data, data_cls, transform=None):
        super().__init__()

        self._data = smol_data
        self.data_cls = data_cls
        self.transform = transform

    @property
    def hparams(self):
        return {}

    @property
    def lengths(self):
        return self._data.seq_length

    def __len__(self):
        return self._data.batch_size

    def __getitem__(self, item):
        molecule = self._data[item]
        if self.transform is not None:

            molecule = self.transform(molecule)

        return molecule

    @classmethod
    @abstractmethod
    def load(cls, data_path, transform=None):
        pass


# *** SmolDataset implementations ***


class GeometricDataset(SmolDataset):
    def sample(self, n_items, replacement=False):
        mol_samples = np.random.choice(
            self._data.to_list(), n_items, replace=replacement
        )
        data = self.data_cls.from_list(mol_samples)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def sample_n_molecules_per_target(self, n_molecules):
        mol_samples = [
            [system for _ in range(n_molecules)] for system in self._data.to_list()
        ]
        mol_samples = list(itertools.chain(*mol_samples))
        data = self.data_cls.from_list(mol_samples)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def split(self, idx, n_chunks):
        chunks = self._data.split(n_chunks)[idx - 1]
        data = self.data_cls.from_list(chunks)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def ddp_split(self, num_replicas: int, rank: int) -> "GeometricDataset":
        data_list = self._data.to_list()
        dataset_size = len(data_list)
        local_data = [data_list[i] for i in range(rank, dataset_size, num_replicas)]
        new_data = self.data_cls.from_list(local_data)
        return GeometricDataset(new_data, self.data_cls, transform=self.transform)

    @classmethod
    def load(
        cls,
        data_path,
        dataset="geom-drugs",
        transform=None,
        remove_hs=False,
        min_size=None,
    ):
        if dataset in ["geom-drugs", "qm9"]:
            data_cls = GeometricMolBatch
        else:
            data_cls = PocketComplexBatch

        if data_path.suffix == ".npz":
            data = load_npz_data(data_path, data_cls, remove_hs)
        else:
            data = load_smol_data(data_path, data_cls, remove_hs)

        if min_size is not None:
            assert dataset in [
                "geom-drugs",
                "qm9",
            ], "min_size filtering for now only supported for geom-drugs and qm9 datasets"
            mols = [mol for mol in data if mol.seq_length >= min_size]
            data = data_cls.from_list(mols)

        return GeometricDataset(data, data_cls, transform=transform)

    def remove(self, indices):
        self._data.remove(indices)

    def append(self, new_data):
        systems = self._data.append(new_data)
        data = PocketComplexBatch(systems)
        return GeometricDataset(data, self.data_cls, transform=self.transform)

    def save(self, save_path, name="train_al", index=0):
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        save_file = save_dir / f"{name}-{index}.smol"
        bytes_data = self._data.to_bytes()
        save_file.write_bytes(bytes_data)

    def save_as_sdf(self, vocab, save_path: str):
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True, parents=True)
        self._data.to_sdf(vocab, save_path=save_path)
