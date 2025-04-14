import os
import pandas as pd

from rdkit import Chem # safe import before ccdc imports
from multiprocessing import Pool, TimeoutError
from ccdc.conformer import ConformerGenerator, ConformerHitList
from ccdc.molecule import Molecule
from ccdc.io import MoleculeWriter, MoleculeReader
from typing import Tuple
from genbench3d.conf_ensemble.conf_ensemble_library import (BIO_CONF_DIRNAME,
                    GEN_CONF_DIRNAME, 
                    DATA_DIRPATH)

class ConfGenerator() :
    """Class to generate conformers for bioactive conformation (in PDBbind)
    
    :param gen_cel_name: Name of the conf ensemble library of generated
        conformers, defaults to 'gen_conf_ensembles'
    :type gen_cel_name: str, optional
    :param root: Data directory
    :type root: str, optional
    """
    
    def __init__(self,
                 gen_cel_name: str = GEN_CONF_DIRNAME,
                 root: str = DATA_DIRPATH) -> None:

        self.gen_cel_name = gen_cel_name
        self.root = root
        self.gen_cel_dir = os.path.join(self.root, self.gen_cel_name)
        if not os.path.exists(self.gen_cel_dir) :
            os.mkdir(self.gen_cel_dir)
    
    
    def generate_conf_for_library(self,
                                  cel_name: str = BIO_CONF_DIRNAME,
                                  ) -> None:
        """Generate conformers for input library

        :param cel_name: Name of the bioactive conformations library
            , defaults to 'pdb_conf_ensembles'
        :type cel_name: str, optional
        """
        
        self.cel_name = cel_name
        self.cel_dir = os.path.join(self.root, self.cel_name)
        cel_df_path = os.path.join(self.cel_dir, 'ensemble_names.csv')
        cel_df = pd.read_csv(cel_df_path)
        params = list(zip(cel_df['ensemble_name'], 
                          cel_df['filename'], 
                          cel_df['smiles']))
            
        # for param in params:
        #     self.generate_conf_thread(param)
            
        with Pool(processes=12, maxtasksperchild=1) as pool :
            iterator = pool.imap(self.generate_conf_thread, params)
            done_looping = False
            while not done_looping:
                try:
                    results = iterator.next(timeout=120) 
                except StopIteration:
                    done_looping = True
                except TimeoutError:
                    print("Generation is too long, returning TimeoutError")
            
        gen_cel_df_path = os.path.join(self.gen_cel_dir, 'ensemble_names.csv')
        cel_df.to_csv(gen_cel_df_path)
            
            
    def generate_conf_thread(self, 
                             params: Tuple[str, str, str]) -> None:
        """
        Thread for multiprocessing Pool to generate confs for a molecule using
        the CCDC conformer generator. Default behaviour is to save all
        generated confs in a dir (and initial+generated ensemble in a 'merged'
        dir in an older version)
        
        :param params: name, SDF filename and SMILES of the molecule
            to generate conformers for
        :type params: Tuple[str, str, str]
        """
        
        name, filename, smiles = params
        # print(smiles)
        gen_file_path = os.path.join(self.gen_cel_dir,
                                         filename)
        if not os.path.exists(gen_file_path):
            try :
                print('Generating for ' + name)
                # ccdc_mol = Molecule.from_string(smiles)
                filepath = os.path.join(self.cel_dir, filename)
                reader = MoleculeReader(filepath)
                ccdc_mol = reader[0]
                
                assert len(ccdc_mol.atoms) > 0
                
                conformers = self.generate_conf_for_mol(ccdc_mol=ccdc_mol)
                
                writer = MoleculeWriter(gen_file_path)
                for conformer in conformers:
                    conformer_mol = conformer.molecule
                    writer.write_molecule(conformer_mol)
                    
            except Exception as e :
                print(f'Generation failed for {name}')
                print(str(e))
        
        return None
        
        
    @staticmethod
    def generate_conf_for_mol(ccdc_mol: Molecule,
                              n_confs: int = 250) -> ConformerHitList:
        """
        Generate conformers for input molecule
        :param ccdc_mol: CCDC molecule to generate confs for
        :type ccdc_mol: Molecule
        :param n_confs: maximum number of confs to generate, defaults to 250
        :type n_confs: int
        :return: confs for molecule
        :rtype: ConformerHitList
        """
        
        conf_generator = ConformerGenerator()
        conf_generator.settings.max_conformers = n_confs
        conformers = conf_generator.generate(ccdc_mol)
        return conformers
    
    
if __name__ == '__main__' :
    cg = ConfGenerator()
    cg.generate_conf_for_library()