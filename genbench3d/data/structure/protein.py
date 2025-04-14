import os
import logging

from MDAnalysis import (Universe, 
                        AtomGroup)
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from openbabel import pybel

class Protein():

    def __init__(self,
                 pdb_filepath: str) -> None:
        assert os.path.exists(pdb_filepath)
        assert pdb_filepath.endswith('.pdb')
        self.pdb_filepath = pdb_filepath
        self._protein_filepath = pdb_filepath.replace('.pdb', 
                                                     '_protein_only.pdb')
        self._protein_clean_filepath = pdb_filepath.replace('.pdb', 
                                                           '_protein_only_clean.pdb')
        
        self._universe = None
        
    
    @property
    def universe(self):
        if self._universe is None:
            self._universe = Universe(self.pdb_filepath)
        return self._universe
        

    @property
    def protein_filepath(self):
        if not os.path.exists(self._protein_filepath):
            self.extract_protein(universe=self.universe,
                                 output_pdb_filepath=self._protein_filepath)
        assert os.path.exists(self._protein_filepath), \
            'Something went wrong during protein extraction'
        return self._protein_filepath
    
    
    @property
    def protein_clean_filepath(self):
        if not os.path.exists(self._protein_clean_filepath):
            self.clean_protein(input_pdb_filepath=self.protein_filepath,
                                 output_pdb_filepath=self._protein_clean_filepath)
        assert os.path.exists(self._protein_clean_filepath), \
            'Something went wrong during protein cleaning'
        return self._protein_clean_filepath
    
    
    def extract_protein(self,
                        universe: Universe,
                        output_pdb_filepath: str) -> None:
        """
        Extracts the protein only from an input pdb file that may also contains
        ligands, cofactors, metals...
        """
        logging.info(f'Extracting protein from universe to {output_pdb_filepath}')
        protein: AtomGroup = universe.select_atoms('protein')
        protein.write(output_pdb_filepath)
        
        
    def extract_ligand(self,
                       universe: str,
                       ligand_name: str,
                       chain: str) -> str:
        """
        Extracts the given ligand from an input pdb file that may also contains
        ligands, cofactors, metals...
        """
        ligand_filepath = self.pdb_filepath.replace('.pdb', 
                                                    f'{ligand_name}_{chain}_ligand_only.pdb')
        logging.info(f"""Extracting ligand {ligand_name} of chain {chain} 
                     from universe to {ligand_filepath}""")
        ligand_selection = f'resname {ligand_name} and segid {chain}'
        ligand: AtomGroup = universe.select_atoms(ligand_selection)
        ligand.write(ligand_filepath)
        return ligand_filepath
        
            
    def clean_protein(self,
                      input_pdb_filepath: str,
                      output_pdb_filepath: str,
                      pH: float = 7.4,
                      ) -> None:
        """
        Adds hydrogens to given pH for a pdb protein (input_filepath)
        """
        logging.info(f'Cleaning protein from {input_pdb_filepath} to {output_pdb_filepath}')
        
        fixer = PDBFixer(filename=input_pdb_filepath)
        # We only use PDBFixer to reinitialize the chains and segment id combinations
        # (e.g. chain A and B, having both segment A and B, will be renamed chain A to D)
        # fixer.findMissingResidues() # it cannot find them with CrossDocked because it does not contain sequence info
        # fixer.findNonstandardResidues()
        # fixer.replaceNonstandardResidues()
        # fixer.findMissingAtoms()
        # fixer.addMissingAtoms()
        # fixer.addMissingHydrogens(pH=pH)
        # fixer.removeHeterogens(keepWater=False)
        intermediate_filepath = output_pdb_filepath.replace('.pdb', '_no_h.pdb')
        PDBFile.writeFile(fixer.topology, fixer.positions, open(intermediate_filepath, 'w'))
            
        # pdbfixer might miss some hydrogens because of missing atoms
        molecule = list(pybel.readfile("pdb", str(intermediate_filepath)))[0]
        # # import pdb;pdb.set_trace()
        # molecule.OBMol.CorrectForPH(pH) # the correct functions fails for some groups: -NH3+ becomes -NH4+
        # molecule.removeh() # some pdb files have hydrogens, these might mess up the next step
        molecule.addh() 
        molecule.write("pdb", str(output_pdb_filepath), overwrite=True) 
        
        
    