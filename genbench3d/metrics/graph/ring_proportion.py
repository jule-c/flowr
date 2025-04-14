from collections import Counter
from typing import Dict

from genbench3d.conf_ensemble import GeneratedCEL

from ..metric import Metric


class RingProportion(Metric):

    def __init__(self, name: str = "Ring proportion") -> None:
        super().__init__(name)

    def get(self, cel: GeneratedCEL, average=True) -> Dict[int, float]:
        all_ring_sizes = []
        for mol in cel.itermols():
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            ring_sizes = [len(ring) for ring in rings]
            all_ring_sizes.extend(ring_sizes)

        self.counter = Counter(all_ring_sizes)
        counter_total = sum(self.counter.values())
        self.value = {k: v / counter_total for k, v in self.counter.items()}
        return self.value
