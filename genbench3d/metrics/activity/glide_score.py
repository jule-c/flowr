import logging
import os
import subprocess
import numpy as np
import pandas as pd

from rdkit import Chem
from genbench3d.conf_ensemble import GeneratedCEL
from ..metric import Metric
from genbench3d.data.structure import GlideProtein

class GlideScore(Metric):
    
    def __init__(self, 
                 glide_protein: GlideProtein,
                 glide_path: str,
                 reference_score: float = 0,
                 name: str = 'Glide score',
                 mininplace: bool = False,
                 ) -> None:
        
        if not os.path.exists(glide_path):
            raise FileNotFoundError(f'Cannot find Glide executable at {glide_path}')
        
        super().__init__(name)
        self.glide_path = glide_path
        self.scores = {}
        self.reference_score = reference_score
        self.glide_protein = glide_protein
        self.mininplace = mininplace
        
        self.glide_output_dirpath = self.glide_protein.glide_output_dirpath
        self.glide_in_filename = 'glide_scoring.in'
        self.glide_in_filepath = os.path.join(self.glide_output_dirpath,
                                              self.glide_in_filename)
        
        self.ligands_filename = 'scored_ligands.sdf'
        self.ligands_filepath = os.path.join(self.glide_output_dirpath,
                                             self.ligands_filename)
        
        self.results_filepath = os.path.join(self.glide_output_dirpath,
                                             'glide_scoring.csv')
        
        
    def get(self, 
            cel: GeneratedCEL) -> list[float]:
        
        with Chem.SDWriter(self.ligands_filepath) as writer:
            i = 0
            glide_names = []
            for name, ce in cel.items():
                mols = ce.to_mol_list()
                for mol in mols:
                    glide_name = f'mol_{i}'
                    i += 1
                    glide_names.append(glide_name)
                    mol.SetProp('_Name', glide_name)
                    writer.write(mol)
                   
        if os.path.exists(self.glide_in_filepath):
            os.remove(self.glide_in_filepath)
        self.generate_glide_in_file() # in case we have different configuration, e.g. inplace and mininplace
                   
        if os.path.exists(self.results_filepath):
            os.remove(self.results_filepath) # Clear before generating new results
                    
        self.run_docking()
        if os.path.exists(self.results_filepath):
            self.scores_df = pd.read_csv(self.results_filepath)
            all_scores = []
            for glide_name in glide_names:
                name_result = self.scores_df[self.scores_df['title'] == glide_name]
                if name_result.shape[0] == 0:
                    logging.info(f'No Glide result for {glide_name}')
                    all_scores.append(np.nan)
                else:
                    all_scores.append(name_result['r_i_docking_score'].values[0])
            
                
            # all_scores = self.scores_df['r_i_docking_score'].values
            
        else:
            logging.warning('Cannot find docking results')
            all_scores = [np.nan] * cel.n_total_confs
        
        # if len(all_scores) != cel.n_total_confs:
        #     import pdb;pdb.set_trace()
        
        return list(all_scores)
    
    
    def generate_glide_in_file(self):
        logging.info(f'Preparing Glide docking input in {self.glide_in_filepath}')
        if self.mininplace:
            docking_method = 'mininplace'
        else:
            docking_method = 'inplace'
        
        d = {'GRIDFILE': self.glide_protein.grid_filepath,
             'OUTPUTDIR': self.glide_output_dirpath,
             'DOCKING_METHOD': docking_method,
             'PRECISION' : 'SP',
             'LIGANDFILE': self.ligands_filepath,
             'POSTDOCK': 'False',
             'NOSORT': 'TRUE',}
        with open(self.glide_in_filepath, 'w') as f:
            for param_name, value in d.items():
                f.write(f'{param_name}   {value}')
                f.write('\n')
                
                
    def run_docking(self):
        
        logging.info(f'Glide docking using {self.glide_in_filepath}')
        command = f'cd {self.glide_output_dirpath} ; {self.glide_path} {self.glide_in_filename} -WAIT -OVERWRITE'
        logging.info(f'Running {command}')
        subprocess.run(command, shell=True)