from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric


class Uniqueness2D(Metric):

    def __init__(self, name: str = "Uniqueness2D") -> None:
        super().__init__(name)
        self.value = None

    def get(self, cel: GeneratedCEL, average=True) -> float:
        if average:
            self.value = cel.n_total_graphs / cel.n_total_confs
        else:
            self.value = cel.n_total_graphs
        return self.value
