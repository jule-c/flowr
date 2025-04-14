from typing import Dict, List

from genbench3d.conf_ensemble import GeneratedCEL

from ..metric import Metric
from .energy_calculator import FFEnergyCalculator


class StrainEnergy(Metric):

    def __init__(
        self, force_field_name: str = "MMFF94s", name: str = "Strain energy"
    ) -> None:
        super().__init__(name)
        self.force_field_name = force_field_name
        self.energy_calculator = FFEnergyCalculator(force_field_name)

        self.value = None
        self.strain_energies: Dict[str, List[float]] = {}

    def get(self, cel: GeneratedCEL, average=True) -> float:
        all_strain_energies = []
        for name, ce in cel.items():
            mol = ce.mol
            mol_strain_energies = []
            for conf in mol.GetConformers():
                conf_id = conf.GetId()
                strain_energy = self.energy_calculator.compute_strain_energy(
                    mol, conf_id
                )
                mol_strain_energies.append(strain_energy)
            self.strain_energies[name] = mol_strain_energies
            all_strain_energies.extend(mol_strain_energies)

        return all_strain_energies
