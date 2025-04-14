from rdkit import Chem

from .data_source import DataSource


class SDFSource(DataSource):

    def __init__(
        self,
        ligands_path: str,
        name: str,
    ) -> None:
        super().__init__(name)
        self.ligands_path = ligands_path

    def __iter__(self):
        for mol in Chem.SDMolSupplier(self.ligands_path):
            yield mol
