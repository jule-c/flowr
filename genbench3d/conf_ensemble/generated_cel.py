from typing import List
from rdkit.Chem import Mol
from .conf_ensemble_library import ConfEnsembleLibrary, BIO_CONF_DIRNAME, DATA_DIRPATH

class GeneratedCEL(ConfEnsembleLibrary):
    
    def __init__(self, 
                 cel_name: str = BIO_CONF_DIRNAME, 
                 root: str = DATA_DIRPATH, 
                 load: bool = True) -> None:
        super().__init__(cel_name, 
                         root,
                         load)
        self.n_total_mols = None
    
    
    @classmethod
    def from_mol_list(cls,
                      mol_list: List[Mol], 
                      cel_name: str = BIO_CONF_DIRNAME,
                      root: str = DATA_DIRPATH,
                      names: List[str] = None,
                      standardize: bool = False,
                      n_total_mols: int = None) -> ConfEnsembleLibrary:
        
        cel = super().from_mol_list(mol_list, 
                                    cel_name, 
                                    root, 
                                    names, 
                                    standardize)
        
        if n_total_mols is not None:
            cel.n_total_mols = n_total_mols
            # allow the number to be higher, 
            # in case all mols were not parsed or saved during generation
        else:
            cel.n_total_mols = len(mol_list)
        return cel
    
    
    # @classmethod
    # def get_cel_subset(cls,
    #                    cel: 'ConfEnsembleLibrary',
    #                    subset_conf_ids: Dict[str, List[int]]):
    #     new_cel = super().get_cel_subset(cel,
    #                                      subset_conf_ids)
    #     return new_cel