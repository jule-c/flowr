import logging
import os

from MDAnalysis import Universe
from openbabel import pybel
from openbabel.pybel import Molecule
from .protein import Protein


class VinaProtein(Protein):
    
    def __init__(self, 
                 pdb_filepath: str,
                 prepare_receptor_bin_path: str) -> None:
        super().__init__(pdb_filepath)
        self.prepare_receptor_bin_path = prepare_receptor_bin_path
        self._pdbqt_filepath = pdb_filepath.replace('.pdb', 
                                                   '.pdbqt')
        
    
    @property
    def pdbqt_filepath(self):
        if not os.path.exists(self._pdbqt_filepath):
            self.vina_prepare_receptor(universe=self.universe,
                                       output_pdbqt_filepath=self._pdbqt_filepath) # Using default configuration
        assert os.path.exists(self._pdbqt_filepath), \
            'Something went wrong during Vina receptor preparation'
        return self._pdbqt_filepath
        
        
    def vina_prepare_receptor(self,
                              universe: Universe,
                              output_pdbqt_filepath: str,
                              ligand_name: str = None,
                                chain: str = None,
                                preparation_method: str = 'adfr',
                                pH: float = 7.4
                                ) -> None:
        """
        inspired from teachopencadd talktorial 15 on protein_ligand_docking
        """
        
        self.extract_protein(universe=universe,
                             output_pdb_filepath=self.protein_filepath)
        
        self.clean_protein(input_pdb_filepath=self.protein_filepath,
                           output_pdb_filepath=self.protein_clean_filepath,
                           pH=pH)
        
        # self.clean_protein(input_pdb_filepath=self.pdb_filepath,
        #                    output_pdb_filepath=self._protein_clean_filepath,
        #                    pH=pH)
        
        if preparation_method == 'adfr':
            self.adfr_receptor_preparation(input_pdb_filepath=self.protein_clean_filepath,
                                           output_pdbqt_filepath=output_pdbqt_filepath)
        else:
            self.pdb_to_pdbqt()
            
        if ligand_name is not None and chain is not None:
            logging.info(f'Extracting ligand data on {ligand_name} and chain {chain}')
            self.current_ligand_filepath = self.extract_ligand(universe=universe,
                                                                ligand_name=ligand_name,
                                                                chain=chain)
        else:
            logging.info('No ligand name and/or chain is given, only computing protein files')
            
            
    def adfr_receptor_preparation(self,
                                  input_pdb_filepath: str,
                                  output_pdbqt_filepath: str,
                                  ) -> None:
        """
        input_pdb_filepath must be a pbd file that only contains the protein with
        hydrogens
        """
        logging.info(f'Preparing protein from {input_pdb_filepath} to {output_pdbqt_filepath}')
        arg_list = [self.prepare_receptor_bin_path,
                    f'-r {input_pdb_filepath}',
                    f'-o {output_pdbqt_filepath}']
        cmd = ' '.join(arg_list)
        os.system(cmd)
        
        
    def pdb_to_pdbqt(self,
                     ph: float = 7.4,
                     ) -> None:
        molecule = list(pybel.readfile("pdb", str(self._protein_filepath)))[0]
        self.ob_mol_to_pdbqt(molecule, ph)
        
        
    def ob_mol_to_pdbqt(self,
                        molecule: Molecule,
                        ph: float = 7.4,
                        ) -> None:
        
        # add hydrogens at given pH
        molecule.OBMol.CorrectForPH(ph)
        molecule.addh()
        # add partial charges to each atom
        for atom in molecule.atoms:
            atom.OBAtom.GetPartialCharge()
        molecule.write("pdb", str(self._protein_clean_filepath), overwrite=True) 
        molecule.write("pdbqt", str(self._pdbqt_filepath), overwrite=True)
        
        # Only keep ATOM and TER lines in pdbqt file
        with open(self.pdbqt_filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        new_lines = [line 
                     for line in lines 
                     if line.startswith('ATOM') or line.startswith('TER')]
        with open(self.pdbqt_filepath, 'w') as f:
            for line in new_lines:
                f.write(line)
                f.write('\n')