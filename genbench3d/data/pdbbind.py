import os
import pandas as pd
import requests
import urllib
import MDAnalysis as mda

from pathlib import Path
# from prody import (parsePDB, 
#                    writePDB)
# from scipy.spatial.distance import euclidean
from collections import Counter
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import Mol
from tqdm import tqdm
from Bio.PDB import PDBParser
from .preprocessing import MolStandardizer
from typing import Union, List, Tuple, Dict, Set
from .ligand_expo import LigandExpo, LigandNameNotInLigandExpo
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from genbench3d.params import (PDBBIND_DIRPATH, 
                    PDBBIND_GENERAL_URL, 
                    PDBBIND_REFINED_URL,
                    PDBBIND_CORE_URL,
                    PDBBIND_GENERAL_DIRPATH,
                    PDBBIND_REFINED_DIRPATH,
                    PDBBIND_CORE_DIRPATH,
                    PDBBIND_GENERAL_TARGZ_FILEPATH,
                    PDBBIND_REFINED_TARGZ_FILEPATH,
                    PDBBIND_CORE_TARGZ_FILEPATH,)

# To be able to save conformer properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)


class PDBIDNotInPDBbindException(Exception) :
    """Raised if the input PDB ID is not in PDBbind
    
    :param pdb_id: Input PDB ID
    :type pdb_id: str
    """
    
    def __init__(self, pdb_id):
        message = f'{pdb_id} is not in PDBbind'
        super().__init__(message)


class LigandExpoFailedParsingException(Exception) :
    
    def __init__(self, pdb_id):
        message = f'RDKit failed to parse PDB ideal ligand SMILES for {pdb_id}'
        super().__init__(message)


class LigandExpoFailedParsingException(Exception) :
    
    def __init__(self, pdb_id):
        message = f'RDKit failed to parse PDB ideal ligand SMILES for {pdb_id}'
        super().__init__(message)


class FailedMatchException(Exception) :
    
    def __init__(self, pdb_id):
        message = f'No match found between actual and ideal ligand for {pdb_id}'
        super().__init__(message)
        
        
class FailedMol2ParsingException(Exception) :
    
    def __init__(self, pdb_id):
        message = f'Failed mol2 parsing for {pdb_id}'
        super().__init__(message)
        
        
class FailedSDFParsingException(Exception) :
    
    def __init__(self, pdb_id):
        message = f'Failed sdf parsing for {pdb_id}'
        super().__init__(message)


class PDBbind() :
    """
    Object to handle PDBbind metadata
    :param root: directory where PDBbind data are located
    :type root: str
    :param corrected: Whether to redownload data (not advised, was done
        for training purposes)
    :type corrected: bool
    :param remove_mers: Whether to remove ligand names ending with '-mer' in PDBbind
    :type remove_mers: bool
    :param remove_unknown_uniprot: Whether to remove complexes with an empty 
        Uniprot accession in the metadata
    :type remove_unknown_uniprot: bool
    :param remove_unknown_ligand_name: Whether to remove complexes with an empty 
        Ligand name in the metadata
    :type remove_unknown_ligand_name: bool
    
    """
    
    def __init__(self, 
                 root: str = PDBBIND_DIRPATH,
                 corrected_files: bool = False,
                 remove_mers: bool = False,
                 remove_unknown_uniprot: bool = False,
                 remove_unknown_ligand_name: bool = False) :
        self.root = root
        self.corrected_files = corrected_files
        self.remove_mers = remove_mers
        self.remove_unknown_uniprot = remove_unknown_uniprot
        self.remove_unknown_ligand_name = remove_unknown_ligand_name
        
        
        if not os.path.exists(PDBBIND_GENERAL_DIRPATH):
            self.download_general_set()
        general_dirname = os.listdir(PDBBIND_GENERAL_DIRPATH)[0]
        self.general_dir_path = os.path.join(PDBBIND_GENERAL_DIRPATH,
                                             general_dirname)
                    
        if not os.path.exists(PDBBIND_REFINED_DIRPATH):
            self.download_refined_set()
        refined_dirname = os.listdir(PDBBIND_REFINED_DIRPATH)[0]
        self.refined_dir_path = os.path.join(PDBBIND_REFINED_DIRPATH,
                                             refined_dirname)
        
        if not os.path.exists(PDBBIND_CORE_DIRPATH):
            self.download_core_set()
        casf2016_dirname = os.listdir(PDBBIND_CORE_DIRPATH)[0]
        self.casf2016_dir_path = os.path.join(PDBBIND_CORE_DIRPATH,
                                             casf2016_dirname)
        core_dirname = 'coreset'
        self.core_dir_path = os.path.join(self.casf2016_dir_path,
                                          core_dirname)
        
        self.pdb_dir = os.path.join(self.root,
                                    'new_pdbs')
        if not os.path.exists(self.pdb_dir) :
            os.mkdir(self.pdb_dir)
        self.ligand_dir = os.path.join(self.root,
                                    'new_ligands')
        if not os.path.exists(self.ligand_dir) :
            os.mkdir(self.ligand_dir)
            
        if self.corrected_files :
            self.available_files = os.listdir(self.ligand_dir)
            self.available_structures = [file.split('.mol2')[0]
                                         for file in self.available_files]
        
        else :
            self.general_available_structures = os.listdir(self.general_dir_path)
            self.refined_available_structures = os.listdir(self.refined_dir_path)
            for filename in ['readme', 'index']:
                self.general_available_structures.remove(filename)
                self.refined_available_structures.remove(filename)
            self.core_available_structures = os.listdir(self.core_dir_path)
            self.available_structures = set(self.general_available_structures
                                            + self.refined_available_structures)
        #self.chembl_targets_df = pd.read_csv('chembl_targets.csv', sep=';')
        self.pdb_ligand_expo = LigandExpo()
        self.mol_standardizer = MolStandardizer()
        
        self.ligand_name_corrections = {
            'A' : 'AMP', # in 5o1u (was done in PDB)
            'MAM' : 'MMA', # in 1ws5 (was done in PDB)
            'GLB' : 'GAL', # in 1pum and 1oko (was done in PDB)
            'BAR' : 'TSA', # in 2cht (was done in PDB)
            'BAM' : 'BEN', # was done in PDB for the 8 ids that are in PDBbind
            'I6P' : 'IHP', # in 5ijj and 5ijp (was done in PDB)
            'U' : 'U5P', # in 1loq and 3gd1 (not done in PDB, but corresponds to single UMP)
            '0IW' : 'D1R', # in 3qsd (not done in PDB, but 0IW is a "biologically interesting molecule" but is same as D1R in PDB)
            } 
        
        self.failed_matches = []
        self.failed_ligand_expo_parsing = []
        self.ligand_name_not_in_ligand_expo = []
        self.failed_pdbbind_parsing = []
        self.failed_mol2_reads = []
        
        
    def download_general_set(self) -> None:
        """Download the general set of PDBbind from the website

        :raises Exception: If the URL is not defined
        """
        
        if PDBBIND_GENERAL_URL is None:
            raise Exception('You need to fill the PDBBIND_GENERAL_URL in params.py')
        else:
            r = requests.get(PDBBIND_GENERAL_URL)
            with open(PDBBIND_GENERAL_TARGZ_FILEPATH, 'wb') as f:
                f.write(r.content)
            os.mkdir(PDBBIND_GENERAL_DIRPATH)
            os.system(f'tar -xzf {PDBBIND_GENERAL_TARGZ_FILEPATH} -C {PDBBIND_GENERAL_DIRPATH}')
                
                
    def download_refined_set(self) -> None:
        """Download the refined set of PDBbind from the website
        """
        
        if PDBBIND_REFINED_URL is None:
            raise Exception('You need to fill the PDBBIND_REFINED_URL in params.py')
        else:
            r = requests.get(PDBBIND_REFINED_URL)
            with open(PDBBIND_REFINED_TARGZ_FILEPATH, 'wb') as f:
                f.write(r.content)
            os.mkdir(PDBBIND_REFINED_DIRPATH)
            os.system(f'tar -xzf {PDBBIND_REFINED_TARGZ_FILEPATH} -C {PDBBIND_REFINED_DIRPATH}')
            
    
    def download_core_set(self) -> None:
        """Download the core set of PDBbind from the website
        """
        
        if PDBBIND_CORE_URL is None:
            raise Exception('You need to fill the PDBBIND_CORE_URL in params.py')
        else:
            r = requests.get(PDBBIND_CORE_URL)
            with open(PDBBIND_CORE_TARGZ_FILEPATH, 'wb') as f:
                f.write(r.content)
            os.mkdir(PDBBIND_CORE_DIRPATH)
            os.system(f'tar -xzf {PDBBIND_CORE_TARGZ_FILEPATH} -C {PDBBIND_CORE_DIRPATH}')
        
        
    def get_master_dataframe(self, 
                             corrections: bool = True
                             ) -> pd.DataFrame:
        """Creates a single dataframe compiling all available metadata
        in PDBbind (PL_data and PL_name files)
        
        :param corrections: Apply a set manual corrections to PDBbind errors.
        :type corrections: bool
        :return: A single pandas DataFrame compiling metadata
        :rtype: pandas.DataFrame
        """
        
        pl_data = self.get_pl_data()
        pl_name = self.get_pl_name()
        
        pl_data['activity_list'] = pl_data['Kd/Ki'].apply(self.parse_activity_string)
        pl_data['sep'] = pl_data['activity_list'].apply(lambda x : x[1])
        pl_data['value'] = pl_data['activity_list'].apply(self.get_nanomolar_activity)
        pl_data['units'] = 'nM'
        pl_all = pl_data.merge(pl_name, on='PDB code')
        active_threshold = 1000 # 1 uM = 1000 nM
        pl_all['active'] = ((pl_all['value'] < active_threshold) 
                            & ~(pl_all['sep'].isin(['>', '~'])))
        
        if self.remove_mers :
            pl_all = pl_all[~(pl_all['ligand name'].str.contains('mer'))]
            
        if corrections:
            pl_all['ligand name'] = pl_all['ligand name'].apply(self.get_corrected_name)
            
        if self.remove_unknown_uniprot:
            pl_all = pl_all[pl_all['Uniprot ID'] != '------']
            
        if self.remove_unknown_ligand_name:
            pl_all = pl_all[pl_all['ligand name'] != '']
            
        self.pl_all = pl_all
        return pl_all
        
        
    def get_corrected_name(self, 
                           name: str
                           ) -> str:
        """
        Correct ligand name
        
        :param name: Ligand name
        :type name: str
        :return: Corrected name
        :rtype: str
        """
        # PDBbind ligand name are mostly circled by parenthesis
        name = name.replace('(', '').replace(')', '')
        if name in self.ligand_name_corrections:
            corrected_name = self.ligand_name_corrections[name]
        else: 
            corrected_name = name
        return corrected_name
        
        
    def get_pl_data(self) -> pd.DataFrame:
        """Compile the PL (protein-ligand) data from PL_data file
        
        :return: A single pandas DataFrame formatting the PL_data file
        :rtype: pandas.DataFrame
        """
        widths = [6,6,7,6,17,9,200]
        cols = ['PDB code',
                'resolution',
                'release year',
                '-logKd/Ki',
                'Kd/Ki',
                'reference',
                'ligand name']
        file_path = os.path.join(self.general_dir_path, 
                                 'index', 
                                 'INDEX_general_PL_data.2020')
        pl_data = pd.read_fwf(file_path, widths=widths, skiprows=6, header=None)
        pl_data.columns=cols
        return pl_data
    
    
    def get_pl_name(self) -> pd.DataFrame:
        """Compile the PL (protein-ligand) data from PL_name file
        
        :return: A single pandas DataFrame formatting the PL_name file
        :rtype: pandas.DataFrame
        """
        widths = [6,6,8,200]
        cols = ['PDB code',
                'release year',
                'Uniprot ID',
                'protein name']
        file_path = os.path.join(self.general_dir_path, 
                                 'index', 
                                 'INDEX_general_PL_name.2020')
        pl_name = pd.read_fwf(file_path, widths=widths, skiprows=6, header=None)
        pl_name.columns=cols
        return pl_name
    
    
    def get_pl_general(self) -> pd.DataFrame:
        """Compile the PL (protein-ligand) data from PL_general file
        
        :return: A single pandas DataFrame formatting the PL_general file
        :rtype: pandas.DataFrame
        """
        widths = [6,6,6,17,9,200]
        cols = ['PDB code', 
                'resolution',
                'release year', 
                'binding data', 
                'reference',
                'ligand name']
        file_path = os.path.join(self.general_dir_path, 
                                 'index', 
                                 'INDEX_general_PL.2020')
        pl_general = pd.read_fwf(file_path, widths=widths,skiprows=6,header=None)
        pl_general.columns=cols
        return pl_general
    
    
    def find_sep_in_activity_string(self, 
                                    string: str, 
                                    possible_seps: List[str] = ['=', '<', '>']
                                    ) -> str:
        """In PDBbind, the activity is represented by a string in Kd/Ki column. 
        This function finds the separator which is the sign in the 
        activity equality.
        
        :param string: Input string from the activity field in master table.
        :type string: str
        :param possible_seps: Possible separators to find in the string. 
            Defaults are "=", "<" or ">".
        :type possible_seps: list[str]
        :return: Found separator. Default is "~" if it does not find any of the
            possible separators
        :rtype: str
        """
        found_sep = '~' # default value
        for sep in possible_seps :
            if sep in string :
                found_sep = sep
                break
        return found_sep

    
    def parse_activity_string(self, 
                              string: str) -> List[str]:
        """Parse the activity (Kd/Ki) string
        Example : "Ki=400mM //" should return ["mM", "=", "400"]
        
        :param string: Activity string to parse
        :type string: str
        :return: A list [unit, sep, value]
        :rtype: list[str, str, str]
        """
        # TODO: Add the type of activity in the return(Kd or Ki or ?)
        
        sep = self.find_sep_in_activity_string(string)
        splitted_string = string.split(sep)
        value_unit = splitted_string[1]

        # maybe a better way to do this with for loop
        parsed = False
        i = 0
        value = ''
        units = ''
        while not parsed :
            char = value_unit[i]
            if char in '0123456789.' :
                value = value + char
            elif char == ' ' :
                parsed = True
            else :
                units = units + char
            i = i + 1
        return [units, sep, value]

    
    def get_nanomolar_activity(self, 
                               l: List[str]) -> float:
        """Get all activity in nanomolar
        
        :param l: list [unit, sep, value] representing activity (i.e. output 
        from parse_activity_string function)
        :type l: list[str, str, str]
        :return: Activity value in nanomolar (nM)
        :rtype: float
        """
        
        unit, sep, value = l
        value = float(value)
        
        # can be done with match case since 3.10
        if unit == 'uM' :
            value = value * 1000
        elif unit == 'mM' :
            value = value * 1000000
        elif unit == 'pM' :
            value = value / 1000
        elif unit == 'fM' :
            value = value / 1000000

        return value

    
    def get_training_test_sets(self, 
                               mode: str = 'ligand',
                               n_classes: int = 50, 
                               train_ratio: float = 0.6,
                               ) -> Tuple[Dict[str, List[str]],
                                          Dict[str, List[str]]]:
        """Performs a train test split per ligand or protein based on the
        complementary binder. For a given ligand/protein name, the list of 
        protein/ligand it binds are stored, reverse sorted by number of 
        occurences, and filling first the training set and then the test set.
        Used in another project.
        
        :param mode: Either ligand or protein
        :type mode: str
        :param n_classes: Number of ligand to include in the dataset 
            (default 50)
        :type n_classes: int
        :param train_ratio: Minimum ratio of samples in the training set
        :type train_ratio: float
        :return: Two dicts train_set and test_set, storing for each ligand key
            a list of pdb_ids
        :rtype: tuple(dict[list], dict[list])
        """
        assert mode in ['ligand', 'protein']
        if mode == 'ligand' :
            class_column_name = 'ligand name'
            binder_column_name = 'Uniprot ID'
        elif mode == 'protein' :
            class_column_name = 'Uniprot ID'
            binder_column_name = 'ligand name'
        
        train_set = {}
        test_set = {}
        pl_all = self.get_master_dataframe()
        if mode == 'protein' :
            pl_all = pl_all[pl_all['Uniprot ID'] != '------']
        class_counts = pl_all[class_column_name].value_counts()
        topN_class_counts = class_counts[:n_classes]
        for class_name in topN_class_counts.index :
            pl_class = pl_all[pl_all[class_column_name] == class_name]
            pl_class = pl_class[pl_class['PDB code'].isin(self.available_structures)]

            # Make sure we have enough data for given class
            if len(pl_class) > 10 : 
                train_pdbs = []
                test_pdbs = []
                counter = Counter()
                counter.update(pl_class[binder_column_name].values)
                if len(counter) > 1 :
                    for binder_name, count in counter.most_common() :
                        pdb_ids = pl_class[pl_class[binder_column_name] == binder_name]['PDB code'].values
                        if len(train_pdbs) < len(pl_class) * train_ratio :
                            train_pdbs.extend(pdb_ids)
                        else :
                            test_pdbs.extend(pdb_ids)
                    train_set[class_name] = train_pdbs
                    test_set[class_name] = test_pdbs
                    
        return (train_set, test_set)
    
    
    def get_pdb_id_pathes(self, 
                          pdb_id: str, 
                          ligand_format: str = 'sdf',
                          ) -> Tuple[str, List[str]]:
        """Give the path to the protein pdb and ligand sdf file(s) for a
        given pdb_id if present in PDBbind
        
        :param pdb_id: Input PDB ID
        :type pdb_id: str
        :param ligand_format: Format of the ligand to return (sdf or mol2)
        :type ligand_format: str
        :return: Tuple with the protein path and the ligand path(es)
        :rtype: tuple(str, list[str])
        """
        
        assert ligand_format in ['sdf', 'mol2'], 'Ligand format is sdf or mol2'
        
        if self.corrected_files :
            
            if pdb_id in self.available_structures :
                protein_path = os.path.join(self.pdb_dir, 
                                            f'{pdb_id}.pdb')
                ligand_path = os.path.join(self.ligand_dir, 
                                           f'{pdb_id}.mol2')
            else :
                raise PDBIDNotInPDBbindException(pdb_id)
            
        else :
            if pdb_id in self.general_available_structures :
                correct_dir_path = self.general_dir_path
            elif pdb_id in self.refined_available_structures :
                correct_dir_path = self.refined_dir_path
            else :
                raise PDBIDNotInPDBbindException(pdb_id)
            
            protein_path = os.path.join(correct_dir_path, 
                                        pdb_id, 
                                        f'{pdb_id}_protein.pdb')
            ligand_path = os.path.join(correct_dir_path, 
                                        pdb_id, 
                                        f'{pdb_id}_ligand.{ligand_format}')
            
        ligand_pathes = [ligand_path]
        return protein_path, ligand_pathes
    
    
    def get_ligand_name(self, 
                        pdb_id: str) -> str:
        """
        Get ligand name from a given PDB ID
        
        :param pdb_id: input PDB ID
        :type pdb_id: str
        :return: Ligand name
        :rtype: str
        """
        
        if not hasattr(self, 'pl_all') :
            self.get_master_dataframe()
            
        # We check if the PDBbind ligand corresponds to the PDB ligand
        pdb_id_info = self.pl_all[self.pl_all['PDB code'] == pdb_id]
        ligand_name = pdb_id_info['ligand name'].values[0]
 
        return ligand_name
    
    
    def get_protein_name(self,
                         uniprot_id: str) -> str:
        """
        Get protein name from a Uniprot accession
        
        :param uniprot_id: input Uniprot accession
        :type uniprot_id: str
        :return: Protein name
        :rtype: str
        """
        master_table = self.get_master_dataframe()
        pdb_lines = master_table[master_table['Uniprot ID'] == uniprot_id]
        protein_names = pdb_lines['protein name'].values
        counter = Counter(protein_names)
        return counter.most_common()[0][0]
    
    
    def get_chains(self,
                   pdb_id: str) -> Set[str]:
        """Get the chains for a given PDB ID

        :param pdb_id: Input PDB ID
        :type pdb_id: str
        :return: Set of chains in the protein 
        :rtype: Set[str]
        """
        
        protein_path, ligand_pathes = self.get_pdb_id_pathes(pdb_id)
        pdb_parser = PDBParser()
        structure = pdb_parser.get_structure('struct', protein_path)
        chains = []
        for model in structure :
            for chain in model :
                chains.append(chain.id)
        return set(chains)
    
    
    def get_ligands(self, 
                    subset: str = 'all',
                    included_pdb_ids: List[str] = None,
                    clean: bool = True) -> List[Mol]:
        """Get all ligands in PDBbind. Can be limited to a subset.

        :param subset: _description_, defaults to 'all', options are ['all',
            'refined', 'general']
        :type subset: str, optional
        :param included_pdb_ids: List of PDB IDs to include, defaults to None
            meaning that all pdb_ids in the dataset are included
        :type included_pdb_ids: List[str], optional
        :param clean: Set to True to perform a cleaning step on each ligands
            , defaults to True
        :type clean: bool, optional
        :return: List of ligands
        :rtype: List[Mol]
        """
        
        self.failed_matches = []
        self.failed_ligand_expo_parsing = []
        self.ligand_name_not_in_ligand_expo = []
        self.failed_pdbbind_parsing = []
        self.failed_mol2_reads = []
        
        assert subset in ['all', 'general', 'refined']
        if subset == 'all' :
            pdb_ids = self.available_structures
        elif subset == 'general' :
            pdb_ids = self.general_available_structures
        elif subset == 'refined' :
            pdb_ids = self.refined_available_structures
        
        table = self.get_master_dataframe()
        pdb_ids = [pdb_id 
                   for pdb_id in pdb_ids
                   if pdb_id in table['PDB code'].values]
            
        if included_pdb_ids is not None: # we select only the included
            pdb_ids = [pdb_id 
                       for pdb_id in pdb_ids
                       if pdb_id in included_pdb_ids]
        
        pdb_ids = sorted(pdb_ids)
        
        mols = []
        for pdb_id in tqdm(pdb_ids) :
            try:
                mol = self.get_ligand(pdb_id, clean)
            except LigandNameNotInLigandExpo:
                self.ligand_name_not_in_ligand_expo.append(pdb_id)
            except FailedSDFParsingException:
                self.failed_pdbbind_parsing.append(pdb_id)
            except LigandExpoFailedParsingException:
                self.failed_ligand_expo_parsing.append(pdb_id)
            except FailedMatchException:
                self.failed_matches.append(pdb_id)
            except Exception as exception:
                # print('Another exception occured')
                print(exception, pdb_id)
            else:
                self.set_conf_prop(mol, pdb_id)
                mols.append(mol)
                
        return mols
    
    
    def get_ligand(self,
                   pdb_id: str,
                   clean: bool = True
                   ) -> Union[Mol, None]:
        """_summary_

        :param pdb_id: PDB ID of the ligand
        :type pdb_id: str
        :param clean: Set to True to clean the ligand, defaults to True
        :type clean: bool, optional
        :raises FailedMol2ParsingException: If mol2 file reading by RDKit leads
            to None, handled in this function
        :raises FailedSDFParsingException: If sdf file reading by RDKit leads
            to None, handled in caller
        :return: Ligand (returns None if cannot be read by mol2 nor sdf file)
        :rtype: Union[Mol, None]
        """
        protein_path, ligand_pathes = self.get_pdb_id_pathes(pdb_id=pdb_id,
                                                                 ligand_format='mol2')
        ligand_path = ligand_pathes[0]
        mol = None
        try :
            mol = self.mol_from_mol2_file(filepath=ligand_path)
            if mol is None :
                raise FailedMol2ParsingException(pdb_id)
            else:
                mol.GetConformer().SetProp('input_format', 'mol2')
                if clean :
                    mol = self.get_clean_ligand(mol, pdb_id)
        except :
            # print('Impossible to read mol2 file for ' + pdb_id)
            
            self.failed_mol2_reads.append(pdb_id)
            
            protein_path, ligand_pathes = self.get_pdb_id_pathes(pdb_id=pdb_id,
                                                                    ligand_format='sdf')
            ligand_path = ligand_pathes[0]
            mol = [m for m in Chem.SDMolSupplier(ligand_path)][0]
            if mol is None :
                raise FailedSDFParsingException(pdb_id)
            else:
                mol.GetConformer().SetProp('input_format', 'sdf')
                if clean :
                    mol = self.get_clean_ligand(mol, pdb_id)
            
        return mol
    
    
    def mol_from_mol2_file(self, 
                           filepath: str) -> Mol :
        """ Get a mol from a mol2 file. Modified such that symbols with more
        than one letter (Cl, Br and As) are only capitalized on the first letter

        :param filepath: Path of the mol2 file
        :type filepath: str
        :return: Molecule
        :rtype: Mol
        """
        with open(filepath, 'r') as f :
            mol2block = f.readlines()  
        mol2block = [line.replace('CL', 'Cl') for line in mol2block]
        mol2block = [line.replace('BR', 'Br') for line in mol2block]
        mol2block = [line.replace('AS', 'As') for line in mol2block]
        mol2block = ''.join(mol2block)
        mol = Chem.MolFromMol2Block(mol2block)
        return mol
    
    
    def set_conf_prop(self,
                      mol: Mol,
                      pdb_id: str) -> None :
        """Set the PDB_ID and pdbbind_id of the main conformer of the molecule
        to the given pdb_id

        :param mol: Input molecule
        :type mol: Mol
        :param pdb_id: PDB ID to set
        :type pdb_id: str
        """
        mol.GetConformer().SetProp('PDB_ID', pdb_id)
        mol.GetConformer().SetProp('pdbbind_id', pdb_id)
    
    
    def get_clean_ligand(self,
                         mol: Mol,
                         pdb_id: str) -> Mol:
        
        """Performs a series of check to clean the ligand.
        Takes the template ligand is present in LigandExpo
        Tries a first stereochemistry-aware matching 
        If no match, tries re-assigning bond orders to match template
        
        
        Check if the PDBbind ligand corresponds to the template given in PDB.
        The ligand must have the same atoms, same chirality.

        :param mol: Input molecule containing conf to clean
        :type mol: Mol
        :param pdb_id: PDB id corresponding to the ligand
        :type pdb_id: str
        :raises LigandExpoFailedParsingException: If there is no template ligand
        :raises FailedMatchException: If no match is possible between the
            template and the actual ligand
        :return: clean mol 
        :rtype: Mol
        """

        ligand_name = self.get_ligand_name(pdb_id)
        
        # Parsing ideal ligand
        ideal_smiles = self.pdb_ligand_expo.get_smiles(ligand_name=ligand_name)
        ideal_mol = Chem.MolFromSmiles(ideal_smiles)
        if ideal_mol is None :
            raise LigandExpoFailedParsingException(pdb_id)
        
        # Sometimes the PDBbind ligand name corresponds to the full ligand
        # And only a part of the ligand is experimentally determined in PDB
        # if mol.GetNumHeavyAtoms() != ideal_mol.GetNumHeavyAtoms() :
        #     print(ligand_name, pdb_id, mol.GetNumHeavyAtoms(), ideal_mol.GetNumHeavyAtoms())
        #     # import pdb; pdb.set_trace()
        #     raise Exception('Ideal ligand and real ligand have different numbers of atom')
        
        # Useful if the ligand is coming from SD file
        Chem.AssignStereochemistryFrom3D(mol)
        
        # Standardize mol with molvs (and neutralization for the ligand)
        
        standard_mol = self.mol_standardizer.standardize(mol)
        standard_ideal = self.mol_standardizer.standardize(ideal_mol, neutralize=False)
        
        # Try a direct match between the 2
        match = standard_ideal.GetSubstructMatch(standard_mol)
        if len(match) != standard_ideal.GetNumHeavyAtoms() :
            # Try using bond order assignement
            standard_mol = AssignBondOrdersFromTemplate(standard_ideal, standard_mol)
            match = standard_ideal.GetSubstructMatch(standard_mol)
            if len(match) != standard_ideal.GetNumAtoms() :
                raise FailedMatchException(pdb_id)
            
        clean_mol = standard_mol
            
        return clean_mol
        
    
    def enumerate_tautomers(self,
                            mol : Mol,
                            keep_stereo: bool = True) -> List[Mol]:
        """ Enumerate tautomers for a molecule

        :param mol: Input molecule
        :type mol: Mol
        :param keep_stereo: Set to True to keep stereochemistry, defaults to True
        :type keep_stereo: bool, optional
        :return: List of tautomers
        :rtype: List[Mol]
        """
        enumerator = rdMolStandardize.TautomerEnumerator()
        if keep_stereo :
            enumerator.SetRemoveBondStereo(False)
            enumerator.SetRemoveSp3Stereo(False)
        tautomers = enumerator.Enumerate(mol)
        tautomers = [tautomer for tautomer in tautomers]
        return tautomers
    
    # Following code could be improved, LigandExpo is used in later project stages
    
    def download_all_corrected_data(self) -> None:
        """Redownload ligands from the current pdb files online
        """
        
        table = self.get_master_dataframe()
        for i, row in tqdm(table.iterrows(), total=table.shape[0]) :
            pdb_id = row['PDB code']
            ligand_name = row['ligand name']
            self.download_corrected_data(pdb_id,
                                         ligand_name)
    
    
    def download_corrected_data(self, 
                                pdb_id: str,
                                ligand_name: str,) :
        """Redownload ligand from the online pdb file

        :param pdb_id: PDB ID to redownload
        :type pdb_id: str
        :param ligand_name: ligand name of interest
        :type ligand_name: str
        """
        
        self.download_pdb_file(pdb_id)
        pdb_path = os.path.join(self.pdb_dir,
                                f'{pdb_id}.pdb')
        # with open(pdb_path, 'r') as f :
        #     lines = f.readlines()
        # for line in lines :
        #     if line.startswith('HETATM') :
        #         ligand_entry = line[17:20].strip()
        #         if ligand_entry == ligand_name :
        #             chain_entry = line[20:22].strip()
        #             res_entry = line[22:26].strip()
                
        #             for ligand_format in ['mol2', 'sdf']:
        #                 self.download_ligand_file(pdb_id, 
        #                                         chain=chain_entry, 
        #                                         res_id=res_entry,
        #                                         ligand_name=ligand_name,
        #                                         ligand_format=ligand_format)
                
        
    def download_pdb_file(self, 
                          pdb_id) :
        pdb_path = os.path.join(self.pdb_dir,
                                f'{pdb_id}.pdb')
        if not os.path.exists(pdb_path) :
            urllib.request.urlretrieve(f'http://files.rcsb.org/download/{pdb_id}.pdb', 
                                       pdb_path)
            
            
    def download_ligand_file(self,
                             pdb_id,
                             chain,
                             res_id,
                             ligand_name,
                             ligand_format) :
        ligand_path = os.path.join(self.ligand_dir,
                                        f'{pdb_id}_{ligand_name}_{chain}_{res_id}.{ligand_format}')
        if not os.path.exists(ligand_path) :
            url = f'https://models.rcsb.org/v1/{pdb_id}/ligand?auth_asym_id={chain}&auth_seq_id={res_id}&encoding={ligand_format}'
            urllib.request.urlretrieve(url, 
                                       ligand_path)
            
            
    def get_pdb_filepath(self,
                         pdb_id: str) -> str:
        return os.path.join(self.pdb_dir, f'{pdb_id}.pdb')
    
    
    # def get_pdb_default_alt_filepath(self,
    #                                  pdb_id: str) -> Union[str, None]:
    #     pdb_filepath = self.get_pdb_filepath(pdb_id)
    #     pdb_default_alt_filepath = pdb_filepath.replace('.pdb', '_default.pdb')
    #     if not os.path.exists(pdb_default_alt_filepath):
    #         atoms = parsePDB(pdb_filepath) # by default, keep altLoc in ['', 'A']
    #         writePDB(pdb_default_alt_filepath, atoms)
    #     if os.path.exists(pdb_default_alt_filepath):
    #         return pdb_default_alt_filepath
    #     else:
    #         return None
        
        
    def get_pocket_filepaths(self,
                            pdb_id: str,
                            keep_altloc: bool = False,
                            ) -> str:
        if keep_altloc:
            pdb_filepath = self.get_pdb_filepath(pdb_id)
        else:
            pdb_filepath = self.get_pdb_default_alt_filepath(pdb_id)
        pocket_filepaths = self.extract_pockets(pdb_id, pdb_filepath)
        return pocket_filepaths
        
    
    def extract_pockets(self,
                       pdb_id: str,
                       pdb_filepath: str,
                       keep_h: bool = False,
                       keep_hetatm: bool = False):
        protein = mda.Universe(pdb_filepath)
        df = self.get_master_dataframe()
        df_row = df[df['PDB code'] == pdb_id]
        ligand_resname = df_row['ligand name'].values[0]

        pocket_filepaths = []
        for residue in protein.residues:
            if residue.resname == ligand_resname:
                ligand_resindex = residue.resindex
                ligand_resid = residue.resid
                ligand_segid = residue.segid

                selections = []
                if not keep_hetatm:
                    selections.append('protein')
                selections.append(f"around 10 resindex {ligand_resindex}")
                if keep_h:
                    selections.append('not type H')
                selection = '(' + ') and ('.join(selections) + ')'
                
                atom_group: mda.AtomGroup = protein.select_atoms(selection)
                if len(atom_group) > 30:
                    # protein_pocket_resids = atom_group.resids
                    # selection = f"resid {' '.join([str(resid) for resid in protein_pocket_resids])}"
                    pocket_resindices = set()
                    for atom in atom_group:
                        resname = atom.resname
                        resindex = atom.resindex
                        pocket_resindices.add(resindex)
                    resindices_str = ' '.join([str(index) 
                                               for index in pocket_resindices])
                    pocket_selection = f'resindex {resindices_str}'
                    protein_pocket: mda.AtomGroup = protein.select_atoms(pocket_selection)
                    pocket_filepath = pdb_filepath.replace('.pdb', 
                                               f'_{ligand_resname}_{ligand_segid}_{ligand_resid}_pocket.pdb')
                    assert '_pocket.pdb' in pocket_filepath
                    pocket_filepaths.append(pocket_filepath)
                    if not os.path.exists(pocket_filepath):
                        protein_pocket.write(pocket_filepath)
                else:
                    print(pdb_filepath, 'Pocket quite small')
        return pocket_filepaths
                    
        
            
            
    # def generate_complex(self,
    #                      pdb_id: str,
    #                      force_write: bool = True) :
    #     """Generate the PDBbind protein-ligand complex based on ligand 
    #     and protein files using pymol

    #     :param pdb_id: Input PDB ID
    #     :type pdb_id: str
    #     :param force_write: Force writing of the complex file, defaults to True
    #     :type force_write: bool, optional
    #     """
    #     protein_path, ligand_pathes = self.get_pdb_id_pathes(pdb_id=pdb_id)
    #     ligand_path = ligand_pathes[0]
    #     complex_path = self.get_complex_path(pdb_id=pdb_id)
    #     if not os.path.exists(complex_path) or force_write :
    #         from pymol import cmd
    #         cmd.set('retain_order', 1)
    #         cmd.load(protein_path)
    #         cmd.load(ligand_path)
    #         cmd.save(complex_path)
    #         cmd.delete('all')
        
        
    # def generate_complexes(self) :
    #     """Generate protein-ligand complexes for all available structures 
    #         in PDBbind
    #     """
    #     for pdb_id in tqdm(self.available_structures) :
    #         self.generate_complex(pdb_id=pdb_id)
            
            
    # def get_complex_path(self, 
    #                      pdb_id: str) -> str:
    #     """Get path of the protein-ligand complex (after generated by the 
    #     generate_complex function)

    #     :param pdb_id: Input PDB ID
    #     :type pdb_id: str
    #     :raises PDBIDNotInPDBbindException: If the pdb_id is not in PDBbind
    #     :return: Path to the complex
    #     :rtype: str
    #     """
    #     if pdb_id in self.general_available_structures :
    #         correct_dir_path = self.general_dir_path
    #     elif pdb_id in self.refined_available_structures :
    #         correct_dir_path = self.refined_dir_path
    #     else :
    #         raise PDBIDNotInPDBbindException(pdb_id)
        
    #     complex_path = os.path.join(correct_dir_path, 
    #                                 pdb_id, 
    #                                 f'{pdb_id}_complex.pdb')
        
    #     return complex_path