import logging
from collections import defaultdict

import numpy as np
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric


class Diversity3D(Metric):

    def __init__(self, name: str = "Diversity3D") -> None:
        super().__init__(name)
        self.value = None

    def get(self, cel: GeneratedCEL, average=True) -> float:
        icds = []
        self.icds = defaultdict(list)
        for name, ce in cel.items():
            try:
                n_confs = ce.mol.GetNumConformers()

                if n_confs > 1:
                    tfd_matrix = cel.get_tfd_matrix(name)
                    if len(tfd_matrix) > 0:
                        icd = np.mean(tfd_matrix)

                        icds.append(icd)
                        self.icds[name].append(icd)
            except Exception as e:
                logging.warning(f"Diversity 3D exception: {e}")

        if len(icds) > 0:
            diversity_3D = np.mean(
                [
                    icd
                    for icd in icds
                    # if icd != 0
                ]
            )
        else:
            diversity_3D = np.nan

        self.value = diversity_3D

        return self.value
