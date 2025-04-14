import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric


class Uniqueness3D(Metric):

    def __init__(self, tfd_threshold: float, name: str = "Uniqueness3D") -> None:
        super().__init__(name)
        self.tfd_threshold = tfd_threshold
        self.value = None

    def get(self, cel: GeneratedCEL, average=True) -> float:
        self.n_unique = 0
        self.unique_mol = {}
        self.clusters = {}
        self.n_tested_confs = 0

        for name, ce in cel.items():

            n_confs = ce.mol.GetNumConformers()

            # if n_confs == 1:
            #     self.n_unique += 1
            # elif n_confs > 1:
            if n_confs > 1:  # we only compute on molecule having multiple conformers
                mol = ce.mol

                try:
                    tfd_matrix = cel.get_tfd_matrix(name)
                    if len(tfd_matrix) > 0:
                        Z = linkage(tfd_matrix)

                        max_value = self.tfd_threshold
                        T = fcluster(Z, t=max_value, criterion="distance")
                        n_clusters = max(T)
                        self.n_unique += n_clusters
                        self.clusters[name] = T

                        if n_clusters == 1 and mol.GetNumConformers() > 1:
                            self.unique_mol[name] = ce.mol
                        self.n_tested_confs += n_confs
                except Exception as e:
                    logging.warning(f"Uniqueness 3D exception: {e}")

            # else:
            #     print('Conf ensemble without conformers, please check')
            #     raise RuntimeError()

        if self.n_unique == 0:
            self.value = np.nan
        else:
            self.value = self.n_unique / self.n_tested_confs

        return self.value
