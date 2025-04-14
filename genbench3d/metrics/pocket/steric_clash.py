from collections import defaultdict

from rdkit import Chem

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.data.structure import Pocket
from genbench3d.geometry import ClashChecker, GeometryExtractor

from ..metric import Metric


class StericClash(Metric):

    def __init__(
        self,
        pocket: Pocket,
        clash_safety_ratio: float,
        consider_hs: bool,
        name: str = "Steric clash",
    ) -> None:
        super().__init__(name)
        self.pocket = pocket
        self.geometry_extractor = GeometryExtractor()
        self.clash_checker = ClashChecker(clash_safety_ratio, consider_hs)
        self.clashes = None
        self.valid_pldist_conf_ids = None

    def get(self, cel: GeneratedCEL, average=True) -> float:

        self.clashes = {}
        self.valid_pldist_conf_ids = defaultdict(list)
        self.n_valid = 0
        all_n_clashes = []

        for name, ce in cel.items():
            ce_clashes = {}

            conf_ids = [conf.GetId() for conf in ce.mol.GetConformers()]
            for conf_id in conf_ids:

                clashes = self.clash_checker.get_pocket_ligand_clashes(
                    pocket=self.pocket, ligand=ce.mol, conf_id=conf_id
                )
                all_n_clashes.append(len(clashes))

                if len(clashes) == 0:
                    self.valid_pldist_conf_ids[name].append(conf_id)
                    self.n_valid += 1
                else:
                    ce_clashes[conf_id] = clashes

            if len(ce_clashes) > 0:
                self.clashes[name] = ce_clashes

        if average:
            self.value = self.n_valid / cel.n_total_confs
        else:
            self.value = self.n_valid

        return all_n_clashes
