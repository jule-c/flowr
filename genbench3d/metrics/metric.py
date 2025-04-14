from abc import ABC, abstractmethod
from rdkit.Chem import Mol
from genbench3d.conf_ensemble import (GeneratedCEL,
                                      ConfEnsembleLibrary)

class Metric(ABC):
    
    def __init__(self,
                 name: str) -> None:
        self.name = name
        self.value = None
        
    @abstractmethod
    def get(self,
            cel: GeneratedCEL) -> float:
        pass
    
    
    def get_from_mol_list(self,
                          mol_list: list[Mol]):
        cel = GeneratedCEL.from_mol_list(mol_list=mol_list)
        return self.get(cel)
    
    
class TrainingMetric(Metric, ABC):
    
    @abstractmethod
    def get(self,
            cel: GeneratedCEL,
            training_cel: ConfEnsembleLibrary) -> float:
        pass