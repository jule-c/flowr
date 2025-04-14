import os
import requests

from rdkit import Chem
from rdkit.Chem import Mol
from .preprocessing import MolStandardizer
from typing import Dict
from genbench3d.params import (LIGANDEXPO_FILEPATH,
                    LIGANDEXPO_URL)

class LigandNameNotInLigandExpo(Exception):
    
    def __init__(self, ligand_name):
        message = f'{ligand_name} is not in Ligand Expo'
        super().__init__(message)


class LigandExpo() :
    """
    Class to extract data from the SMILES file from PDB Ligand expo (dictionnary)
    where the 'standard' SMILES of each ligand in PDB is stored
    http://ligand-expo.rcsb.org/dictionaries/Components-smiles-stereo-cactvs.smi
    
    :param smiles_filepath: Path to the file containing the SMILES
    :type smiles_filepath: str
    
    """

    def __init__(self,
                 smiles_filepath = LIGANDEXPO_FILEPATH) :
        self.smiles_filepath = smiles_filepath
        if not os.path.exists(self.smiles_filepath):
            self.download_ligand_expo()
        
        self.smiles_d = self._get_smiles_d() # Links a PDB ligand name to corresponding smiles in cactvs
        self.mol_standardizer = MolStandardizer()
        
    def download_ligand_expo(self) -> None:
        """Download the ligand expo smiles file
        """
        r = requests.get(LIGANDEXPO_URL)
        # import pdb;pdb.set_trace()
        with open(LIGANDEXPO_FILEPATH, 'wb') as f:
            f.write(r.content)
        
        
    def _get_smiles_d(self) -> Dict[str, str]:
        """
        Return a dictionnary mapping each ligand PDB identifier to SMILES
        
        :return: Dictionnary linking PDB ligand id to SMILES
        :rtype: Dict[str, str]
        
        """
        d = {}
        with open(self.smiles_filepath, 'r') as f :
            lines = f.readlines() # a line is SMILES\tLIGANDID\tLIGANDFULLNAME
        for line in lines :
            l = line.strip().split('\t')
            if len(l) == 3 : # there might be lines having no smiles
                smiles = l[0]
                ligand_name = l[1]
                d[ligand_name] = smiles
        return d
    
    def get_smiles(self,
                   ligand_name: str) -> str:
        """
        Return the SMILES for a given ligand name
        
        :param ligand_name: Ligand identifier in PDB
        :type ligand_name: str
        :return: Corresponding SMILES
        :rtype: str
        """
        smiles = None
        if ligand_name in self.smiles_d : 
            smiles = self.smiles_d[ligand_name]
        else :
            raise LigandNameNotInLigandExpo(ligand_name)
        return smiles
    
    def get_standard_ligand(self,
                            ligand_name: str,
                            standardize: bool = True) -> Mol:
        """
        Return the standardized RDKit mol for a given ligand name
        
        :param ligand_name: Ligand identifier in PDB
        :type ligand_name: str
        :param standardize: Set to True to standardize the ligand
        :type standardize: bool
        :return: Corresponding standard mol
        :rtype: Mol
        """
        smiles = self.get_smiles(ligand_name)
        mol = Chem.MolFromSmiles(smiles)
        if standardize:
            standard_mol = self.mol_standardizer.standardize(mol, neutralize=False)
        else:
            standard_mol = mol
        return standard_mol