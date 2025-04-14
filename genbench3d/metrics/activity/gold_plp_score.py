import logging
import numpy as np

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
from rdkit import Chem
from rdkit.Chem import Mol
from .gold_scorer import GoldScorer

class GoldPLPScore(Metric):
    
    def __init__(self, 
                 protein_path: str,
                 native_ligand: Mol,
                 name: str = 'Gold PLP score',
                 ) -> None:
        super().__init__(name)
        self.protein_path = protein_path
        self.gold_scorer = GoldScorer(protein_path,
                                      native_ligand)
        self.plp_scores = {}
        
        
    def get(self, 
            cel: GeneratedCEL) -> float:
        try:
            mols = cel.to_mol_list()
            if len(mols) > 0:
                all_scores = self.gold_scorer.score_mols(mols)
            else:
                all_scores = []
        except Exception as e:
            logging.warning(f'Gold Docking issue: {e}')
            all_scores = [np.nan] * cel.n_total_confs
        
        # if len(all_scores) != cel.n_total_confs:
        #     import pdb;pdb.set_trace()
        
        # all_scores = []
        # for name, ce in cel.items():
        #     mols = ce.to_mol_list()
        #     scores = self.gold_scorer.score_mols(mols)
        #     self.plp_scores[name] = scores
        #     all_scores.extend(scores)
        
        return all_scores