import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.data.structure import Pocket
from genbench3d.metrics import Metric


class DistanceToNativeCentroid(Metric):

    def __init__(
        self,
        pocket: Pocket,
        name: str = "Distance to native centroid",
    ) -> None:
        super().__init__(name)
        self.pocket = pocket
        self.distances = None
        self.valid_pldist_conf_ids = None

    def get(self, cel: GeneratedCEL, average=True) -> float:

        self.distances = {}

        native_ligand = self.pocket.native_ligand
        native_conf = native_ligand.GetConformer()
        native_centroid = self._get_centroid(native_conf)

        all_distances = []
        for name, ce in cel.items():
            mol = ce.mol
            mol_distances = []
            for conf in mol.GetConformers():
                centroid = self._get_centroid(conf)
                distance_to_cendroid = np.linalg.norm(centroid - native_centroid)
                mol_distances.append(distance_to_cendroid)

            self.distances[name] = mol_distances
            all_distances.extend(mol_distances)

        return all_distances

    @staticmethod
    def _get_centroid(conf: Conformer):
        pos = conf.GetPositions()
        centroid = pos.mean(axis=0)
        return centroid
