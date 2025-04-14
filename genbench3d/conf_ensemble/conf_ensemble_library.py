import logging
import os
from copy import deepcopy
from typing import Dict, Generator, ItemsView, Iterable, KeysView, List, ValuesView

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from scipy.spatial.distance import squareform
from tqdm import tqdm

from genbench3d.utils import get_full_matrix_from_tril

from ..utils.molconfviewer import MolConfViewer
from .conf_ensemble import ConfEnsemble

# These are legacy behaviour from previous work when CEL needed to be saved for faster processing
BIO_CONF_DIRNAME = "generated_molecules"
GEN_CONF_DIRNAME = "generated_molecules"
DATA_DIRPATH = "."


class ConfEnsembleLibrary:
    """
    Class to store multiple molecules having each multiple confs. The backend
    is a dict having the ensemble names as keys.
    ConfEnsembleLibrary = CEL

    :param cel_name: name of the default directory to store ensembles in. In this
        work, it stores bioactive conformations from PDBBind
    :type cel_name: str
    :param root: data directory
    :type root: str

    """

    def __init__(
        self,
        cel_name: str = BIO_CONF_DIRNAME,
        root: str = DATA_DIRPATH,
        load: bool = True,
    ) -> None:
        self.root = root
        self.cel_name = cel_name
        self.cel_dir = os.path.join(self.root, self.cel_name)
        self._library: Dict[str, ConfEnsemble] = {}
        self.csv_filename = "ensemble_names.csv"
        self.csv_filepath = os.path.join(self.cel_dir, self.csv_filename)
        if load:
            self.load()

        self._morgan_fps = None
        self._tfd_matrices = {}

    @property
    def cel_df(self):
        if not hasattr(self, "_cel_df"):
            if not os.path.exists(self.csv_filepath):
                self.save()
            else:
                self.load()
        return self._cel_df

    @property
    def n_total_confs(self):
        return sum([mol.GetNumConformers() for mol in self.itermols()])

    @property
    def n_total_graphs(self):
        return len(self)

    def __len__(self) -> int:
        return len(self._library)

    def __iter__(self) -> Iterable[ConfEnsemble]:
        return iter(self._library.values())

    def __contains__(self, name: str) -> bool:
        return name in self._library

    def __getitem__(self, name: str) -> ConfEnsemble:
        return self._library[name]

    def __setitem__(self, name: str, ce: ConfEnsemble) -> None:
        self._library[name] = ce

    def get(self, name: str) -> ConfEnsemble:
        return self[name]

    def get_mol(self, name: str) -> Mol:
        ce = self[name]
        return ce.get_mol()

    def set(self, name: str, conf_ensemble: ConfEnsemble) -> None:
        assert isinstance(conf_ensemble, ConfEnsemble)
        self._library[name] = conf_ensemble

    def keys(self) -> KeysView:
        return self._library.keys()

    def names(self) -> list[str]:
        return list(self.keys())

    def values(self) -> ValuesView:
        return self._library.values()

    def items(self) -> ItemsView:
        return self._library.items()

    def itermols(self) -> Generator[Mol, None, None]:
        for ce in self._library.values():
            yield ce.get_mol()

    @classmethod
    def from_mol_list(
        cls,
        mol_list: List[Mol],
        cel_name: str = BIO_CONF_DIRNAME,
        root: str = DATA_DIRPATH,
        names: List[str] = None,
        standardize: bool = False,
    ) -> "ConfEnsembleLibrary":
        """
        Constructor to create a library from a list of molecules (containing one
        or more confs).

        :param mol_list: list of input molecules
        :type mol_list: List[Mol]
        :param cel_name: name of the directory where the library will be stored
        :type cel_name: str
        :param root: data directory
        :type root: str
        :param names: list of ensemble name for each molecule, to be used to group
            in ensembles
        :type names: List[str]
        :param standardize: Set to True if you want the molecules to be
            standardized with molvs
        :type standardized: bool

        """
        for mol in mol_list:
            assert isinstance(mol, Mol)

        conf_ensemble_library = cls(cel_name, root, load=False)

        if names is None:
            names = []
            unknown_counter = 0
            for mol in mol_list:
                try:
                    name = Chem.MolToSmiles(mol)
                except Exception:
                    print("Unknown SMILES")
                    name = f"Unknown_{unknown_counter}"
                    unknown_counter += 1
                names.append(name)
        else:
            assert len(mol_list) == len(
                names
            ), "mol_list and names should have the same length"

        for name, mol in zip(names, mol_list):
            if name not in conf_ensemble_library:
                ce = ConfEnsemble(mol_list=[mol], name=name, standardize=standardize)
                conf_ensemble_library[name] = ce
            else:
                ce = conf_ensemble_library.get(name)
                ce.add_mol(mol, standardize=standardize)

        return conf_ensemble_library

    @classmethod
    def from_mol_dict(
        cls,
        mol_dict: Dict[str, List[Mol]],
        cel_name: str = BIO_CONF_DIRNAME,
        root: str = DATA_DIRPATH,
        standardize: bool = False,
        renumber_atoms: bool = False,
    ) -> "ConfEnsembleLibrary":
        """
        Creates a ConfEnsemble from a dict of molecules each containing conformations

        :param mol_dict: dict of RDKit molecules
        :type mol_dict: Dict[str, Mol]
        :param cel_name: name of the library
        :type cel_name: str
        :param root: root directory of data storage
        :type root: str
        :param standardize: Set to True if you want the molecules to be
            standardized with molvs
        :type standardized: bool
        :param renumber_atoms: Set True if molecules do not have the same atom order
            and require to match atoms between template and new molecules
        :type renumber_atoms: bool

        """
        conf_ensemble_library = cls(cel_name, root, load=False)
        for name, mol_list in tqdm(mol_dict.items()):
            try:
                ce = ConfEnsemble(
                    mol_list=mol_list,
                    name=name,
                    standardize=standardize,
                    renumber_atoms=renumber_atoms,
                )
                conf_ensemble_library[name] = ce
            except Exception as e:
                print(f"conf ensemble failed for {name}")
                print(str(e))
        # conf_ensemble_library.save()
        return conf_ensemble_library

    @classmethod
    def from_ce_dict(
        cls,
        ce_dict: Dict[str, ConfEnsemble],
        cel_name: str = BIO_CONF_DIRNAME,
        root: str = DATA_DIRPATH,
        standardize: bool = False,
    ) -> "ConfEnsembleLibrary":
        """
        Creates a ConfEnsembleLibrary from a dict of ConfEnsemble

        :param mol_dict: dict of ConfEnsemble
        :type mol_dict: Dict[str, ConfEnsemble]
        :param cel_name: name of the library
        :type cel_name: str
        :param root: root directory of data storage
        :type root: str

        """
        conf_ensemble_library = cls(cel_name, root)
        for name, ce in tqdm(ce_dict.items()):
            try:
                conf_ensemble_library[name] = ce
            except Exception as e:
                print(f"conf ensemble failed for {name}")
                print(str(e))
        # conf_ensemble_library.save()
        return conf_ensemble_library

    def load(self) -> None:
        """
        Load the ConfEnsembles in the cel_dir given as input when creating the library
        and load associated metadata in cel_df

        """
        if os.path.exists(self.csv_filepath):
            self._cel_df = pd.read_csv(self.csv_filepath)
            print("Loading conf ensembles")
            for i, row in tqdm(self._cel_df.iterrows()):
                name = row["ensemble_name"]
                filename = row["filename"]
                filepath = os.path.join(self.cel_dir, filename)
                try:
                    ce = ConfEnsemble.from_file(filepath, name)
                    self._library[name] = ce
                except:
                    print(f"Loading failed for {name}")

    def save(self) -> None:
        """
        Save the ConfEnsembles in the cel_dir, and the metadata in cel_df and pdbbind_df

        """
        if not os.path.exists(self.cel_dir):
            os.mkdir(self.cel_dir)
        names = list(self.keys())
        print("Saving conf ensembles")
        self._cel_df = pd.DataFrame(columns=["ensemble_name", "smiles", "filename"])
        for name_i, name in enumerate(tqdm(names)):
            writer_filename = f"{name_i}.sdf"
            writer_path = os.path.join(self.cel_dir, writer_filename)
            ce = self.get(name)
            ce.save_ensemble(sd_writer_path=writer_path)
            smiles = Chem.MolToSmiles(ce.mol)
            row = pd.DataFrame(
                [[name, smiles, writer_filename]], columns=self._cel_df.columns
            )
            self._cel_df = pd.concat([self._cel_df, row], ignore_index=True)
        self._cel_df.to_csv(self.csv_filepath, index=False)
        self.create_pdb_df()

    def merge(self, new_conf_ensemble_library: "ConfEnsembleLibrary") -> None:
        """
        Merge the ConfEnsembles from the input library and current library
        Only add conformations to molecules existing in the current library

        :param conf_ensemble_library: Input library to add confs from
        :type conf_ensemble_library: ConfEnsembleLibrary
        """
        for name in self.keys():
            if name in new_conf_ensemble_library.keys():
                ce = self.get(name)
                new_mol = new_conf_ensemble_library.get_mol(name)
                try:
                    ce.add_mol(mol=new_mol)
                except Exception as e:
                    print(f"Merging didn't work for {name}: ", e)
            else:
                print(f"{name} is not in the second library")

    # we could use the PDBbind class to do that
    def create_pdb_df(self) -> None:
        """
        Create the DataFrame associating each ligand name to its pdb id
        for the current library
        """
        pdb_df = pd.DataFrame(columns=["ligand_name", "pdb_id"])
        for name, ce in tqdm(self.items()):
            mol = ce.get_mol()
            for conf in mol.GetConformers():
                pdb_id = conf.GetProp("PDB_ID")
                row = pd.DataFrame([[name, pdb_id]], columns=pdb_df.columns)
                pdb_df = pd.concat([pdb_df, row], ignore_index=True)
        pdb_df_path = os.path.join(self.cel_dir, "pdb_df.csv")
        pdb_df.to_csv(pdb_df_path, index=False)

    @staticmethod
    def view_ensemble(
        name: str, cel_name: str = GEN_CONF_DIRNAME, root: str = DATA_DIRPATH
    ):
        """
        View the ConfEnsemble for the input molecule name from the
        given cel_name using MolConfViewer

        :param name: Name of the molecule
        :type name: str
        :param cel_name: Name of the library to refer to
        :type cel_name: str
        :param root: Data directory
        :type root: str

        """
        cel_dir = os.path.join(root, cel_name)
        csv_filename = "ensemble_names.csv"
        csv_filepath = os.path.join(cel_dir, csv_filename)
        cel_df = pd.read_csv(csv_filepath)
        try:
            filename = cel_df[cel_df["ensemble_name"] == name]["filename"].values[0]
            filepath = os.path.join(cel_dir, filename)
            ce = ConfEnsemble.from_file(filepath, name)
            mol = ce.get_mol()
            viewer = MolConfViewer()
            viewer.view(mol)
        except Exception as e:
            print(str(e))
            print(f"Loading failed for {name}")

    # TODO: change to cel_names (allow more than 2 CE)
    @classmethod
    def get_merged_ce(
        cls,
        filename: str,
        name: str,
        root: str = DATA_DIRPATH,
        cel_name1: str = BIO_CONF_DIRNAME,
        cel_name2: str = GEN_CONF_DIRNAME,
        embed_hydrogens: bool = False,
    ) -> ConfEnsemble:
        """
        Merge the ConfEnsemble from 2 libraries for a molecule. We assume
        that the filename is the same for both libraries (e.g. bioactive and
        generated conformations of the same molecules in 2 different libraries)

        :param filename: Name of the file (i.sdf)
        :type filename: str
        :param name: Name of the molecule
        :type name: str
        :param root: Data directory
        :type root:str
        :param cel_name1: Name of the first library (e.g. containing bioactive conformations)
        :type cel_name1: str
        :param cel_name1: Name of the second library (e.g. containing generated conformations)
        :type cel_name1: str
        :param embed_hydrogens: Set to True to keep hydrogens in loaded molecules
        :type embed_hydrogens: bool
        :return: ConfEnsemble with conformations from the 2 libraries for the molecule
        :rtype: ConfEnsemble
        """
        cel_dir1 = os.path.join(root, cel_name1)

        ce_filepath = os.path.join(cel_dir1, filename)
        conf_ensemble = ConfEnsemble.from_file(
            filepath=ce_filepath, name=name, embed_hydrogens=embed_hydrogens
        )

        cel_dir2 = os.path.join(root, cel_name2)
        gen_ce_filepath = os.path.join(cel_dir2, filename)
        gen_conf_ensemble = ConfEnsemble.from_file(
            filepath=gen_ce_filepath, name=name, embed_hydrogens=embed_hydrogens
        )

        new_mol = conf_ensemble.get_mol()
        gen_conf_ensemble.add_mol(new_mol, standardize=False)
        return gen_conf_ensemble

    def select_smiles_list(self, smiles_list: List[str]) -> None:
        """Keep only given smiles in the library, and remove all the others

        :param smiles_list: List of smiles to keep in the library
        :type smiles_list: List[str]
        """
        subset_df = self._cel_df[self._cel_df["smiles"].isin(smiles_list)]
        subset_names = subset_df["ensemble_name"].unique()

        names_to_remove = []
        for ensemble_name in self.keys():
            if ensemble_name in subset_names:
                names_to_remove.append(ensemble_name)

        for ensemble_name in names_to_remove:
            self._library.pop(ensemble_name)

        self._cel_df = subset_df.copy().reset_index(drop=True)

    @property
    def morgan_fps(self):
        if self._morgan_fps is None:
            self._morgan_fps = self.compute_morgan_fps()
        return self._morgan_fps

    def compute_morgan_fps(self) -> List[ExplicitBitVect]:
        morgan_fps = []
        for mol in self.itermols():
            mol = Chem.RemoveHs(mol)
            morgan_fp = GetMorganFingerprintAsBitVect(
                mol=mol, radius=3, useChirality=True
            )
            morgan_fps.append(morgan_fp)
        return morgan_fps

    def get_tfd_matrix(self, name: str) -> List[float]:
        if not name in self._tfd_matrices:
            self._tfd_matrices[name] = self.compute_tfd_matrix(name)
        return self._tfd_matrices[name]

    def compute_tfd_matrix(self, name: str) -> List[float]:
        tfd_matrix = []
        try:
            ce = self[name]
            mol = ce.mol
            tfd_matrix = GetTFDMatrix(mol)
            n_confs = mol.GetNumConformers()
            tfd_matrix = get_full_matrix_from_tril(tfd_matrix, n=n_confs)
            tfd_matrix = squareform(tfd_matrix)  # to get to triu
        except Exception as e:
            logging.warning(e)
        return tfd_matrix

    @classmethod
    def get_cel_subset(
        cls, cel: "ConfEnsembleLibrary", subset_conf_ids: Dict[str, List[int]]
    ):
        new_cel = cls(cel.cel_name, cel.root, load=False)

        for name, conf_ids in subset_conf_ids.items():
            if name in cel:
                ce = cel[name]
                if len(conf_ids) > 0:
                    new_ce = deepcopy(ce)
                    original_conf_ids = [
                        conf.GetId() for conf in ce.mol.GetConformers()
                    ]
                    for conf_id in original_conf_ids:
                        if conf_id not in conf_ids:
                            new_ce.mol.RemoveConformer(conf_id)
                    new_cel[name] = new_ce
            else:
                print("Unknown name, please check")
                import pdb

                pdb.set_trace()

        return new_cel

    def to_mol_list(self):
        mols = []
        for ce in self:
            ce_mols = ce.to_mol_list()
            mols.extend(ce_mols)
        return mols
