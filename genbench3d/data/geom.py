import json
import pickle
import os
import random

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Mol
from genbench3d.params import GEOM_DRUGS_SUMMARY_FILEPATH, GEOM_RDKIT_DIRPATH
from typing import Dict

class GEOMDrugs():
    
    def __init__(self,
                 root: str = GEOM_RDKIT_DIRPATH,
                 summary_filepath: str = GEOM_DRUGS_SUMMARY_FILEPATH) -> None:
        self.root = root
        with open(summary_filepath, 'r') as f:
            self.summary = json.load(f)
            
        
    def get_mols(self,
                ) -> Dict[str, Mol]:
        
        mol_dict = {}
        smiles_list = list(self.summary.keys())
        random.shuffle(smiles_list)
        for smiles in tqdm(smiles_list[:5000]):
            d = self.summary[smiles]
            if 'pickle_path' in d:
                pickle_path = d['pickle_path']
                pickle_filepath = os.path.join(self.root,
                                            pickle_path)
                with open(pickle_filepath, 'rb') as f:
                    geom_entry = pickle.load(f)
                    
                mols = []
                wrong_smiles = False
                for geom_conformer in geom_entry['conformers']:
                    rd_mol = geom_conformer['rd_mol']
                    # achiral_mol = Mol(rd_mol)
                    # Chem.RemoveStereochemistry(achiral_mol)
                    # achiral_mol = Chem.RemoveHs(rd_mol)
                    # current_smiles = Chem.MolToSmiles(achiral_mol)
                    try:
                        current_smiles = Chem.MolToSmiles(Chem.RemoveHs(rd_mol), 
                                                        isomericSmiles=False)
                        input_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles),
                                                        isomericSmiles=False)
                        if current_smiles != input_smiles:
                            wrong_smiles = True
                            # import pdb;pdb.set_trace()
                            break
                    except:
                        wrong_smiles = True
                        # import pdb;pdb.set_trace()
                        break
                    conf = rd_mol.GetConformer()
                    prop_names = list(geom_conformer.keys())
                    prop_names.remove('rd_mol')
                    for prop_name in prop_names:
                        value = geom_conformer[prop_name]
                        if type(value) == float:
                            conf.SetDoubleProp(prop_name, value)
                        elif type(value) == str:
                            conf.SetProp(prop_name, value)
                        elif type(value) == int:
                            conf.SetIntProp(prop_name, value)
                        elif type(value) == list:
                            value = str(value)
                            conf.SetProp(prop_name, value)
                        else:
                            raise Exception('Unknown type, please define which function needs to be used')
                    
                    mols.append(rd_mol)
                    
                if not wrong_smiles:
                    mol_dict[smiles] = mols
            
        # cel = ConfEnsembleLibrary.from_mol_dict(mol_dict=mol_dict,
        #                                         cel_name='GEOMDrugs_CEL',
        #                                          standardize=False,
        #                                          renumber_atoms=False)
        
        return mol_dict
            
    