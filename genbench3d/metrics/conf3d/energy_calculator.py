import logging
import numpy as np

from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from typing import Tuple, Union


class FFEnergyCalculator():
    
    def __init__(self,
                 force_field_name: str = 'MMFF94s'):
        assert force_field_name in ['MMFF94s', 'MMFF94', 'UFF'], \
            'Please select a valid force field name among [MMFF94s, MMFF94, UFF]'
        self.force_field_name = force_field_name
        
    def compute_strain_energy(self,
                              mol: Mol,
                              conf_id: int = 0,
                              return_mol: bool = False,
                              n_steps: int = 1000) -> Union[Tuple[float, Mol],
                                                                 float]:
        new_mol = Mol(mol)
        new_mol = Chem.AddHs(new_mol, addCoords=True)
        
        try:
        
            if self.force_field_name == 'MMFF94s':
                mol_properties = AllChem.MMFFGetMoleculeProperties(new_mol, 
                                                                    mmffVariant='MMFF94s')
                force_field = AllChem.MMFFGetMoleculeForceField(new_mol, 
                                                                mol_properties,
                                                                confId=conf_id)
            elif self.force_field_name == 'MMFF94':
                mol_properties = AllChem.MMFFGetMoleculeProperties(new_mol)
                force_field = AllChem.MMFFGetMoleculeForceField(new_mol, 
                                                                mol_properties,
                                                                confId=conf_id)
            elif self.force_field_name == 'UFF':
                force_field = AllChem.UFFGetMoleculeForceField(new_mol, confId=conf_id)
            
            start_energy = force_field.CalcEnergy()
            not_converged = force_field.Minimize(maxIts=n_steps)
            if not_converged:
                print('Not converged')
            final_energy = force_field.CalcEnergy()
            strain_energy = start_energy - final_energy
            assert strain_energy > 0
            
        except Exception as e:
            
            logging.warning('Force field error')
            strain_energy = np.nan
            
        if return_mol:
            return strain_energy, new_mol
        else:
            return strain_energy