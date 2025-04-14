from .data_source import DataSource
from rdkit import Chem

class LigBoundConf(DataSource):
    
    def __init__(self,
                 ligands_path: str,
                 name: str = 'LigBoundConf',
                 ) -> None:
        super().__init__(name)
        self.ligands_path = ligands_path
    
    def __iter__(self):
        for mol in Chem.SDMolSupplier(self.ligands_path):
            yield mol