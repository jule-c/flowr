import os
import time
from typing import List, Union

from rdkit import Chem
from rdkit.Chem import SDWriter
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFMCS import FindMCS

from ..data.preprocessing.mol_standardizer import MolStandardizer


class ConfEnsemble:
    """
    Class able to store different confs for the same molecule in a single
    rdkit molecule. The workflow was made to standardize input molecules,
    check if they are the same, and add conformations with atom matching using
    rdkit molecule atom ordering

    :param mol_list: List of identical molecules but different confs
    :type mol_list: List[Mol]
    :param name: name of the ensemble. Default is template molecule smiles
    :type name: str
    :param template_mol: Molecule to serve as template, all new molecule to add
        will match this template (useful when you have a single 2D template)
    :type template_mol: Mol
    :param standardize: Uses molvs to standardize the molecular graph if True
        (careful as this step removes hydrogens if input contains hydrogens)
    :type standardize: bool
    :param renumber_atoms: Set True if molecules do not have the same atom order
        and require to match atoms between template and new molecules
    :type renumber_atoms: bool

    """

    def __init__(
        self,
        mol_list: List[Mol],
        name: str = None,
        template_mol: Mol = None,
        standardize: bool = True,
        renumber_atoms: bool = True,
    ) -> None:
        self.name = name

        if template_mol is None:
            template_mol = Mol(mol_list[0])
            mol_list = mol_list[1:]

        if name is None:
            smiles = Chem.MolToSmiles(template_mol)
            name = smiles

        if standardize:
            self._mol_standardizer = MolStandardizer()
            template_mol = self._mol_standardizer.standardize(
                template_mol, neutralize=False
            )
        self._mol = template_mol

        for mol in mol_list:
            self.add_mol(mol, standardize, renumber_atoms)

        self._mol.SetProp("_Name", name)

    @property
    def mol(self):
        return self._mol

    def get_mol(self):
        return self.mol

    def add_mol(
        self, mol: Mol, standardize: bool = True, renumber_atoms: bool = True
    ) -> None:
        """
        Add molecule to the ensemble. Molecule must be the same (graph) as the
        template, only conformers should be different

        :param mol: Molecule to add
        :type mol: Mol
        :param standardize: Uses molvs to standardize the molecular graph if True
            (careful as this step removes hydrogens if input contains hydrogens)
        :type standardize: bool
        :param renumber_atoms: Set True if molecules do not have the same atom order
            and require to match atoms between template and new molecules
        :type renumber_atoms: bool

        """

        if standardize:
            standard_mol = self._mol_standardizer.standardize(mol)
        else:
            standard_mol = mol

        if renumber_atoms:
            try:
                mcs = FindMCS([self._mol, mol])
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                match = standard_mol.GetSubstructMatch(mcs_mol)
                if len(match) != self._mol.GetNumHeavyAtoms():
                    # Try using bond order assignement
                    standard_mol = AssignBondOrdersFromTemplate(self._mol, standard_mol)
                    mcs = FindMCS([self._mol, mol])
                    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
                    match = standard_mol.GetSubstructMatch(mcs_mol)

                self_match = self._mol.GetSubstructMatch(mcs_mol)
                assert len(match) == len(self_match)
                self2mcs = {res: idx for idx, res in enumerate(self_match)}
                self2new = {idx: match[res] for idx, res in self2mcs.items()}
                new_match = []
                for i in range(len(self2new)):
                    new_match.append(self2new[i])
                # new_match: nparray such that new_match[self_mol_index] = new_mol_index

                # reverse_self_match = {i_mcs: i_self for i_self, i_mcs in enumerate(self_match)}
                # new_match = [reverse_self_match[i_mcs] for i_mcs in match]
                renumbered_mol = Chem.RenumberAtoms(standard_mol, new_match)
            except Exception:
                renumbered_mol = standard_mol
        else:
            renumbered_mol = standard_mol

        for new_conf in renumbered_mol.GetConformers():
            # conf_id = new_conf.GetId()
            # original_conf = standard_mol.GetConformer(conf_id)
            # prop_names = original_conf.GetPropNames(includePrivate=True,
            #                                         includeComputed=True)
            # for prop in prop_names:
            #     value = original_conf.GetProp(prop)
            #     new_conf.SetProp(prop, str(value))
            self._mol.AddConformer(new_conf, assignId=True)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        name: str = None,
        standardize: bool = False,
        renumber_atoms: bool = False,
        embed_hydrogens: bool = True,
        output_type: str = "conf_ensemble",
    ) -> Union["ConfEnsemble", List[Mol]]:
        """
        Constructor to create a conf ensemble from a molecule file. Accepted
        file formats are sdf and mol2

        :param filepath: path to the sdf file containing multiple confs of the
        same molecule
        :type filepath: str
        :param name: name of the ensemble
        :type name: str
        :param standardize: Uses molvs to standardize the molecular graph if True
            (careful as this step removes hydrogens if input contains hydrogens)
        :type standardize: bool
        :param renumber_atoms: Set True if molecules do not have the same atom order
            and require to match atoms between template and new molecules
        :type renumber_atoms: bool
        :param embed_hydrogens: Set True if you want the conformer ensemble to
        keep hydrogens, set to False if you want to remove hydrogens
        :type embed_hydrogens: bool
        :param output_type: Type of the output of this function. Default
        behaviour is to produce a ConfEnsemble. Setting to mol_list returns the
        molecule list (useful for debug purposes)
        :type output_type: str

        :return:

        """

        assert output_type in [
            "conf_ensemble",
            "mol_list",
        ], "Output type must be either conf_ensemble or mol_list"

        file_format = filepath.split(".")[-1]
        assert file_format in ["sdf", "mol2"], "File format must be sdf or mol2"

        removeHs = not embed_hydrogens

        if file_format == "sdf":
            with open(filepath, "rb") as f:
                suppl = Chem.ForwardSDMolSupplier(f, removeHs=removeHs)
                mol_list = [mol for mol in suppl]
        elif file_format == "mol2":
            mol_list = [Chem.MolFromMol2File(filepath)]

        def mol_props_to_conf(mol):
            prop_names = mol.GetPropNames(includePrivate=True)
            for prop_name in prop_names:

                add_prop = True
                if "Gold" in prop_name:
                    if not prop_name == "Gold.PLP.Fitness":
                        add_prop = False

                if add_prop:
                    value = mol.GetProp(prop_name)
                    for conf in mol.GetConformers():
                        conf.SetProp(prop_name, value)

        for mol in mol_list:
            mol_props_to_conf(mol)

        if name is None:
            name = Chem.MolToSmiles(mol_list[0])

        if output_type == "conf_ensemble":
            ce = cls(
                mol_list, name, standardize=standardize, renumber_atoms=renumber_atoms
            )
            return ce
        else:
            return mol_list

    def save_ensemble(self, sd_writer_path: str) -> None:
        """
        Save the ensemble in an SDF file

        :param sd_writer_path: SDF file path to store all confs for the molecule
        :type sd_writer_path: str

        """

        sd_writer = SDWriter(sd_writer_path)
        self.save_confs_to_writer(writer=sd_writer)

    def save_confs_to_writer(self, writer: SDWriter) -> None:
        """
        Save each conf of the RDKit mol as a single molecule in a SDF

        :param writer: RDKit writer object
        :type writer: SDWriter

        """
        mol = Mol(self._mol)
        for conf in mol.GetConformers():
            conf_id = conf.GetId()
            # Store the conf properties as molecule properties to save them
            prop_names = conf.GetPropNames(includePrivate=True, includeComputed=True)
            for prop in prop_names:
                value = conf.GetProp(prop)
                mol.SetProp(prop, str(value))
            writer.write(mol=mol, confId=conf_id)

    def to_mol_list(self):
        mols = []
        # conf_ids = [conf.GetId() for conf in self._mol.GetConformers()]
        # for conserved_conf_id in conf_ids:
        #     new_mol = Mol(self._mol)
        #     for erased_conf_id in conf_ids:
        #         if conserved_conf_id != erased_conf_id:
        #             new_mol.RemoveConformer(erased_conf_id)
        #     mols.append(new_mol)
        mols = [
            Mol(mol=self._mol, confId=conf.GetId())
            for conf in self._mol.GetConformers()
        ]
        return mols
