import warnings
from typing import List, Union

import numpy as np
from MDAnalysis import AtomGroup
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import Mol
from vina import Vina

from genbench3d.data.structure import VinaProtein

warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="WARNING: Search space volume is greater than 27000 Angstrom^3 (See FAQ)",
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VinaScorer:

    def __init__(
        self,
        vina_protein: VinaProtein,
        sf_name: str,
        n_cpus: int,
        seed: int,
        size_border: float,
    ) -> None:

        self.vina_protein = vina_protein
        self.sf_name = sf_name
        self.n_cpus = n_cpus
        self.seed = seed
        self.size_border = size_border

        self._vina = Vina(sf_name=sf_name, cpu=n_cpus, seed=seed, verbosity=0)
        self._vina.set_receptor(self.vina_protein.pdbqt_filepath)
        # will automatically perform protein preparation if
        # the pdbqt file does not exist

        # at this stage, the Vina box is not defined
        # it is recommended to use the class methods to instantiate this class

        # if not os.path.exists(VINA_BIN_FILEPATH):
        #     self.download_vina()

    # @staticmethod
    # def download_vina() -> None:
    #     if not os.path.exists(VINA_DIRPATH):
    #         os.mkdir(VINA_DIRPATH)
    #     os.system(f'wget {VINA_URL} -O {VINA_BIN_FILEPATH}')

    @classmethod
    def from_ligand(
        cls,
        ligand: Union[Mol, AtomGroup],
        vina_protein: VinaProtein,
        sf_name: str,
        n_cpus: int,
        seed: int,
        size_border: float,
    ) -> "VinaScorer":
        vina_scorer = cls(vina_protein, sf_name, n_cpus, seed, size_border)
        vina_scorer.set_box_from_ligand(ligand, size_border=size_border)
        return vina_scorer

    @classmethod
    def from_ligand_name_chain(
        cls,
        ligand_name: str,
        chain: str,
        vina_protein: VinaProtein,
        sf_name: str,
        n_cpus: int,
        seed: int,
        size_border: float,
    ) -> "VinaScorer":
        vina_scorer = cls(vina_protein, sf_name, n_cpus, seed, size_border)
        ligand_filepath = vina_scorer.vina_protein.extract_ligand(
            universe=vina_scorer.protein.universe, ligand_name=ligand_name, chain=chain
        )
        ligand = Chem.MolFromPDBFile(ligand_filepath)
        vina_scorer.set_box_from_ligand(ligand, size_border=size_border)
        return vina_scorer

    def set_box_from_ligand(self, ligand: Union[Mol, AtomGroup], size_border: float):

        # with open(self.box_filepath, 'r') as f:
        #     lines: List[str] = [line.strip() for line in f.readlines()]
        # d = {}
        # for line in lines:
        #     l = line.split(' ')
        #     key = l[0]
        #     value = l[2]
        #     d[key] = float(value)
        # center = [d[f'center_{axis}'] for axis in 'xyz']
        # box_size = [d[f'size_{axis}'] for axis in 'xyz']

        if isinstance(ligand, Mol):
            conf = ligand.GetConformer()
            ligand_positions = conf.GetPositions()
        elif isinstance(ligand, AtomGroup):
            ligand_positions = ligand.positions
        else:
            import pdb

            pdb.set_trace()
            raise Exception("Input ligand should be RDKit Mol or MD Universe")
        center = (ligand_positions.max(axis=0) + ligand_positions.min(axis=0)) / 2
        # box_size = ligand_positions.max(axis=0) - ligand_positions.min(axis=0) + size_border
        box_size = [size_border] * 3
        self._vina.compute_vina_maps(center=center, box_size=box_size)

    # def write_box_from_ligand(self,
    #                              ligand: Union[Mol, AtomGroup],
    #                              size_border: float = DEFAULT_SIZE_BORDER, # Angstrom
    #                              ) -> None:
    #     with open(self.box_filepath, 'w') as f:
    #         for center_value, axis in zip(box_center, 'xyz'):
    #             f.write(f'center_{axis} = {center_value}')
    #             f.write('\n')
    #         for size_value, axis in zip(box_size, 'xyz'):
    #             f.write(f'size_{axis} = {size_value}')
    #             f.write('\n')

    def score_mol(
        self,
        ligand: Mol,
        minimized: bool = False,
        output_filepath: str = None,
        add_hydrogens: bool = True,
    ) -> List[float]:

        energies = self.score_mols(
            ligands=[ligand],
            minimized=minimized,
            output_filepath=output_filepath,
            add_hydrogens=add_hydrogens,
        )
        return energies

    # Doesn't work for len(ligands) > 1 for some reason...
    def score_mols(
        self,
        ligands: List[Mol],
        minimized: bool = False,
        output_filepath: str = None,
        add_hydrogens: bool = True,
    ) -> List[float]:

        assert (
            self._vina._center is not None
        ), "You need to setup the pocket box using the set_box_from_ligand function"

        # Ligand preparation
        if add_hydrogens:
            ligands = [Chem.AddHs(ligand, addCoords=True) for ligand in ligands]

        pdbqt_strings = []
        ligand_names = []
        for i, ligand in enumerate(ligands):
            ligand_name = f"Ligand_{i}"
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(ligand)
            mol_setup = mol_setups[0]
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                setup=mol_setup, bad_charge_ok=True
            )
            if is_ok:
                pdbqt_strings.append(pdbqt_string)
                ligand_names.append(ligand_name)

        # import pdb;pdb.set_trace()

        if len(pdbqt_strings) > 0:
            self._vina.set_ligand_from_string(pdbqt_strings)
            scores = []
            if minimized:
                energies = self._vina.optimize()
            else:
                energies = self._vina.score()

            if output_filepath is not None:
                self._vina.write_pose(output_filepath, overwrite=True)

            if len(pdbqt_strings) > 1:
                current_pos = 0
                for i in range(len(ligands)):
                    ligand_name = f"Ligand_{i}"
                    if ligand_name in ligand_names:
                        scores.append(energies[current_pos][0])
                        current_pos += 1
                    else:
                        scores.append(np.nan)
                scores = [energy[0] for energy in energies]
            else:
                scores = energies[:1]
        else:
            scores = [np.nan] * len(ligands)
        return scores

    def score_pdbqt(
        self,
        pdbqt_filepath: str,
        minimized: bool = False,
    ) -> List[float]:

        assert (
            self._vina._center is not None
        ), "You need to setup the pocket box using the set_box_from_ligand function"
        assert pdbqt_filepath.endswith("pdbqt")

        self._vina.set_ligand_from_file(pdbqt_filepath)
        if minimized:
            energies = self._vina.optimize()
        else:
            energies = self._vina.score()
        return energies

    def score_sdf(
        self,
        sdf_filepath: str,
    ) -> List[List[float]]:

        assert (
            self._vina._center is not None
        ), "You need to setup the pocket box using the set_box_from_ligand function"
        assert sdf_filepath.endswith("sdf")

        with Chem.SDMolSupplier(sdf_filepath) as suppl:
            mols = [mol for mol in suppl]

        all_energies = []
        for ligand in mols:
            energies = self.score_mol(ligand)
            all_energies.append(energies)

        return all_energies

    def score_pdb(
        self,
        pdb_filepath: str,
        minimized: bool = False,
    ) -> List[float]:

        assert (
            self._vina._center is not None
        ), "You need to setup the pocket box using the set_box_from_ligand function"
        assert pdb_filepath.endswith("pdb")

        ligand = Chem.MolFromPDBFile(pdb_filepath)
        energies = self.score_mol(ligand, minimized)

        return energies
