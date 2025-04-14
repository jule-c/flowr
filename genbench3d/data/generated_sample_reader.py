import tempfile
import logging
import os
import numpy as np

from tqdm import tqdm
from abc import abstractmethod
from ase.db import connect
from ase.io import write
from openbabel import openbabel as ob
from openbabel import pybel
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import (rdDetermineBonds,
                        Mol)
from typing import List, Tuple
from genbench3d.params import DATA_DIRPATH

# TEMPDIR = tempfile.gettempdir()
# TEMPFILE_NAME = 'mol.sdf'
# TEMPFILE_PATH = os.path.join(TEMPDIR, TEMPFILE_NAME)
POSSIBLE_CHARGES = [0]
for i in range(1, 10):
    POSSIBLE_CHARGES.extend([i, -i])


class BondDeterminationFail(Exception):
    
    def __init__(self) -> None:
        self.message = 'Bond determination failed'
        super().__init__(self.message)


class GeneratedSampleReader():
    
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def read(self) -> List[Mol]:
        pass
    

class ASEDBReader(GeneratedSampleReader):
    
    def __init__(self,
                 bond_determination_backend: str = 'rdkit') -> None:
        super().__init__()
        assert bond_determination_backend in ['babel', 'rdkit']
        self.bond_determination_backend = bond_determination_backend
    
    def read(self, 
             filepath: str,
             root: str = DATA_DIRPATH) -> Tuple[List[Mol], int]:
        
        assert isinstance(cel_name, str), \
            'You must provide a valid ConfEnsembleLibrary name'
        
        sampled_mols = []
        
        db = connect(filepath)
        logging.info('Reading ASE database')
        
        n_mols = len(db)
        for i in tqdm(range(n_mols)):
            try:
                idx = i + 1
                atom_row = db[idx]
                atoms = atom_row.toatoms()
                if self.bond_determination_backend == 'babel':
                    ob_mol = self.babel_mol_from_ase_atoms(atoms)
                    mol2_block = ob_mol.write(format='mol2', 
                            overwrite=True)
                    rd_mol = Chem.MolFromMol2Block(mol2_block)
                elif self.bond_determination_backend == 'rdkit':
                    rd_mol = self.rd_mol_from_ase_atoms(atoms)
                else:
                    raise Exception('Backend is not correct')
                if rd_mol is None:
                    raise BondDeterminationFail()
                    
                sampled_mols.append(rd_mol)
            except Exception as e:
                print(e)
                print(str(e))
        
        return sampled_mols, n_mols
            
            
    @staticmethod
    def babel_mol_from_ase_atoms(atoms: Atoms):
        
        positions = atoms.positions
        numbers = atoms.numbers
        
        obmol = ob.OBMol()
        obmol.BeginModify()
        for p, n in zip(positions, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
            
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        mol = pybel.Molecule(obmol)
        
        return mol
    
    
    @staticmethod
    def rd_mol_from_ase_atoms(atoms: Atoms,
                              possible_charges: List[int] = POSSIBLE_CHARGES,
                              full_check: bool = False): # full_check doesnt work to remove of radical electrons
        # write(TEMPFILE_PATH, atoms)
        xyz_block = ASEDBReader.xyz_block_from_ase_atoms(atoms)
        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        conn_mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        current_mol = None
        for charge in possible_charges:
            try:
                rdDetermineBonds.DetermineBondOrders(conn_mol,charge=charge)
            except:
                # import pdb;pdb.set_trace()
                # bonds_are_determined = False
                pass # Charge is not correct, retrying with another charge
            else:
                if current_mol is None:
                    current_mol = Chem.Mol(conn_mol)
                    if not full_check:
                        break
                    current_n_rads = sum([atom.GetNumRadicalElectrons() 
                                        for atom in conn_mol.GetAtoms()])
                else:
                    n_rads = sum([atom.GetNumRadicalElectrons() 
                                for atom in conn_mol.GetAtoms()])
                    if n_rads < current_n_rads:
                        current_mol = Chem.Mol(conn_mol)
                        current_n_rads = n_rads
                # break # Charge is correct, we can return conn_mol
        return current_mol
            
        
    @staticmethod
    def xyz_block_from_ase_atoms(atoms: Atoms,
                                 comment: str = '',
                                 fmt='%22.15f'):
        # inspired from the write_xyz function in ase, instead it writes a string
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/io/xyz.html#write_xyz
        xyz_block = ''
        natoms = len(atoms)
        xyz_block = xyz_block + '%d\n%s\n' % (natoms, comment)
        for s, (x, y, z) in zip(atoms.symbols, atoms.positions):
            xyz_block = xyz_block + '%-2s %s %s %s\n' % (s, fmt % x, fmt % y, fmt % z)
        return xyz_block