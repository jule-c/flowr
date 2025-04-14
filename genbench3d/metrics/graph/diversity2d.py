import numpy as np
from rdkit.DataStructs import TanimotoSimilarity

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric


class Diversity2D(Metric):

    def __init__(self, name: str = "Diversity2D") -> None:
        super().__init__(name)
        self.value = None
        self.dists = None

    def get(self, cel: GeneratedCEL, average=True) -> float:
        self.dists = []
        morgan_fps = cel.morgan_fps
        n_mols = cel.n_total_graphs
        for i in range(n_mols):
            fp1 = morgan_fps[i]
            for j in range(i + 1, n_mols):
                fp2 = morgan_fps[j]
                sim = TanimotoSimilarity(fp1, fp2)
                dist = 1 - sim
                self.dists.append(dist)

        if average:
            diversity = np.nanmean(self.dists)
        else:
            diversity = np.sum(self.dists)
        return float(diversity)
