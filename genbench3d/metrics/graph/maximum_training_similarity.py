import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity

from genbench3d.conf_ensemble import ConfEnsembleLibrary, GeneratedCEL
from genbench3d.metrics import TrainingMetric


class MaxTrainSim(TrainingMetric):

    def __init__(self, name: str = "Maximum training similarity") -> None:
        super().__init__(name)
        self.value = None
        self.max_sims = None

    def get(
        self, cel: GeneratedCEL, training_cel: ConfEnsembleLibrary, average=True
    ) -> float:
        all_max_sims = []
        self.max_sims = {}
        for name, fp in zip(cel.keys(), cel.morgan_fps):
            if name in training_cel.keys():
                max_sim = 1
            else:
                sims = BulkTanimotoSimilarity(fp, training_cel.morgan_fps)
                max_sim = np.max(sims)

                # check if 2 fps are the same even though smiles is not
                # if max_sim == 1:
                #     print(name)
                #     print(np.argmax(sims))

            all_max_sims.append(max_sim)
            self.max_sims[name] = max_sim

        # import pdb;pdb.set_trace()

        return all_max_sims
