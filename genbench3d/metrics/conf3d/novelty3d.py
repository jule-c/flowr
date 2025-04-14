import logging
from collections import defaultdict

import numpy as np
from rdkit.Chem.TorsionFingerprints import GetTFDBetweenMolecules

from genbench3d.conf_ensemble import ConfEnsembleLibrary, GeneratedCEL
from genbench3d.metrics import TrainingMetric


class Novelty3D(TrainingMetric):

    def __init__(self, tfd_threshold: float, name: str = "Novelty3D") -> None:
        super().__init__(name)
        self.tfd_threshold = tfd_threshold
        self.value = None
        self.n_novel = None
        self.novel_conf_ids = None

    def get(
        self, cel: GeneratedCEL, training_cel: ConfEnsembleLibrary, average=True
    ) -> float:

        self.n_novel = 0
        self.novel_conf_ids = defaultdict(list)

        self.n_tested_confs = 0
        for name, ce in cel.items():

            ce = cel[name]

            # n_confs = ce.mol.GetNumConformers()
            # if not name in training_cel:
            #     self.n_novel += n_confs
            # else:
            if name in training_cel:
                mol = ce.mol

                training_ce = training_cel[name]
                training_mol = training_ce.mol

                for conf1 in mol.GetConformers():
                    self.n_tested_confs += 1
                    conf_id1 = conf1.GetId()
                    tfds = []
                    is_novel = False
                    for conf2 in training_mol.GetConformers():
                        conf_id2 = conf2.GetId()
                        try:
                            tfd = GetTFDBetweenMolecules(
                                mol1=mol,
                                mol2=training_mol,
                                confId1=conf_id1,
                                confId2=conf_id2,
                            )
                            tfds.append(tfd)
                        except Exception as e:
                            logging.warning(f"Novelty 3D exception: {e}")
                    if len(tfds) > 0:  # When tfds is empty: not novel
                        is_novel = min(tfds) > self.tfd_threshold
                    if is_novel:
                        self.novel_conf_ids[name].append(conf_id1)
                        self.n_novel += 1

        if self.n_tested_confs == 0:
            self.value = np.nan
        else:
            if average:
                self.value = self.n_novel / self.n_tested_confs
            else:
                self.value = self.n_novel

        return self.value
