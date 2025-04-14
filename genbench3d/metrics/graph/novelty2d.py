from genbench3d.conf_ensemble import ConfEnsembleLibrary, GeneratedCEL
from genbench3d.metrics import TrainingMetric


class Novelty2D(TrainingMetric):

    def __init__(self, name: str = "Novelty2D") -> None:
        super().__init__(name)
        self.value = None
        self.novel_names = None
        self.n_novel = None

    def get(
        self, cel: GeneratedCEL, training_cel: ConfEnsembleLibrary, average=True
    ) -> float:
        self.novel_names = [
            name for name in cel.keys() if name not in training_cel.keys()
        ]
        self.n_novel = len(self.novel_names)
        if average:
            self.value = self.n_novel / cel.n_total_graphs
        else:
            self.value = self.n_novel

        return self.value
