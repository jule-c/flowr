import logging
import os
import sys
import warnings

from genbench3d.conf_ensemble import GeneratedCEL

from ..metric import Metric
from .vina_scorer import VinaScorer

warnings.filterwarnings(
    "ignore",
    category=Warning,
    message="WARNING: Search space volume is greater than 27000 Angstrom^3 (See FAQ)",
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


class VinaScore(Metric):

    def __init__(
        self,
        vina_scorer: VinaScorer,
        name: str = "Vina score",
        minimized: bool = False,
    ) -> None:
        super().__init__(name)
        self.scores = {}
        self.vina_scorer = vina_scorer
        self.minimized = minimized

    # def get(self,
    #         cel: GeneratedCEL) -> float:
    #     all_scores = []
    #     mols = cel.to_mol_list()
    #     for mol in mols:
    #         try:
    #             # start_time = time.time()
    #             all_scores = self.vina_scorer.score_mols(ligands=mols,
    #                                                 minimized=self.minimized)
    #             if all_scores is None:
    #                 raise Exception('Failed molecule preparation')
    #             # logging.info(f'Time: {time.time() - start_time}')
    #         except Exception as e:
    #             logging.warning(f'Vina scoring error: {e}')

    #     if len(all_scores) != cel.n_total_confs:
    #         import pdb;pdb.set_trace()

    #     return all_scores

    def get(self, cel: GeneratedCEL) -> float:
        all_scores = []
        for name, ce in cel.items():
            ce_scores = []
            mols = ce.to_mol_list()
            for mol in mols:
                try:
                    # start_time = time.time()
                    scores = self.vina_scorer.score_mol(
                        ligand=mol, minimized=self.minimized
                    )
                    if scores is None:
                        raise Exception("Failed molecule preparation")
                    # logging.info(f'Time: {time.time() - start_time}')
                    score = scores[0]
                    ce_scores.append(score)
                except Exception as e:
                    logging.warning(f"Vina scoring error: {e}")
            self.scores[name] = ce_scores
            all_scores.extend(ce_scores)

        # if len(all_scores) != cel.n_total_confs:
        #     import pdb;pdb.set_trace()

        return all_scores
