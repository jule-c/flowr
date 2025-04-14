import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import urllib
import matplotlib.image as mpimg
import os

from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Lipinski import RotatableBondSmarts

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg, GetDihedralDeg
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN
from typing import List, Dict, Tuple
from scipy.stats import wasserstein_distance
#from scipy.spatial.distance import squareform
    

class ConfScorer(object) :
    
    def __init__(self, rmsd_func='rdkit'):
        self.analyser = ConfAnalyzer()
        self.rmsd_func = rmsd_func
        self.rmsd_thresholds = [0.5, 1, 1.25, 1.5, 2]
        
        # Have a memory to avoid recomputing results, useful for large molecules
        self.keys = []
        self.analyzed_molecule_pairs = []
        self.coverages = []
        self.matchings = []
        
    def get_interconformer_rmsds(self, rdmol: Mol) -> np.ndarray :
        """
        Returns the array of RMSD between unique pairs of different conformers
        Args :
            rdmol: Mol = RDKit molecule containing the conformers to compare
        Returns :
            np.ndarray = array of pairwise RMSD between each conformers
        """
        
        rmsd_matrix = self._get_rmsd_matrix(rdmol)
        unique_indices = np.tril_indices(rmsd_matrix.shape[0], -1)
        rmsds = rmsd_matrix[unique_indices]
        
        return rmsds
        
        
    def get_interconformer_mean_rmsd(self, rdmol: Mol) -> float :
        """
        Args :
            rdmol: Mol = RDKit molecule containing the conformers to compare
        Returns :
            float = Mean of interconformer RMSDs
        """
        
        rmsds = self.get_interconformer_rmsds(rdmol)
        return rmsds.mean()
    
    
    def get_interconformer_std_rmsd(self, rdmol: Mol) -> float :
        """
        Args :
            rdmol: Mol = RDKit molecule containing the conformers to compare
        Returns :
            float = Standard deviation of interconformer RMSDs
        """
        
        rmsds = self.get_interconformer_rmsds(rdmol)
        return rmsds.std()
        
        
    def compare_conformation_libraries(self,
                                       ref_conf_lib: Dict[str, Mol],
                                       gen_conf_lib: Dict[str, Mol],
                                       reset_memory: bool = True,
                                       verbose: bool = True,
                                       remove_hydrogens = True
                                      ) -> None :
        """
        Compare 2 libraries containing molecules with different sets of conformation.
        Typically a reference library and a generated library.
        Keys are usually smiles, but might be more complicated (comparing the
        conformations for a smiles in different conditions i.e. target conditioned generation)
        Save and print evaluation metrics
        Args :
            ref_conf_lib: Dict[str, Mol] = Dictionnary of {key : Mol} with conformations
                embedded in the Mol
            gen_conf_lib: Dict[str, Mol] = Dictionnary of {key : Mol} with conformations
                embedded in the Mol
            reset_memory: bool = reset the memory of computed pairs of reference and generated
                conformations (default = True)
        """
        
        reference_keys = set(ref_conf_lib.keys())
        generated_keys = set(ref_conf_lib.keys())
        assert reference_keys == generated_keys
        
        if reset_memory :
            self.keys = []
            self.analyzed_molecule_pairs = []
            self.coverages = []
            self.matching = []
        
        for key in reference_keys :
            
            if verbose :
                print(key)
            
            self.keys.append(key)
            ref_rdmol = ref_conf_lib[key]
            gen_rdmol = gen_conf_lib[key]
            
            if remove_hydrogens :
                ref_rdmol = Chem.RemoveHs(ref_rdmol)
                gen_rdmol = Chem.RemoveHs(gen_rdmol)
            
            if Chem.MolToSmiles(ref_rdmol) == Chem.MolToSmiles(gen_rdmol) :
                coverages, matching = self.get_coverage_matching(ref_rdmol, gen_rdmol, remove_hydrogens=False)
            else :
                print(f'The reference and generated are not the same molecule for key {key}')
              
        all_coverages = defaultdict(list)
        for mol_coverages in self.coverages :
            for threshold in self.rmsd_thresholds :
                all_coverages[threshold].append(mol_coverages[threshold])
                
        for threshold in self.rmsd_thresholds :
            print(f'Threshold = {threshold}')
            print(f'Mean coverage (%) = {np.mean(np.array(all_coverages[threshold]) * 100)}')
            print(f'Median coverage (%) = {np.median(np.array(all_coverages[threshold]) * 100)}')
        
        print(f'Mean matching = {np.mean(np.array(self.matchings))}')
        print(f'Median matching = {np.median(np.array(self.matchings))}')
        
        
    def get_coverage_matching(self, 
                              ref_rdmol: Mol, 
                              gen_rdmol: Mol,
                              rmsd_thresholds: list = None,
                              remove_hydrogens: bool = True
                             ) -> Tuple[dict, float] :
        
        """ 
        Each reference conformation has a closest generated conformation, with a minimum RMSD.
        The coverage is the ratio of reference conformations having a minimum RMSD below a given threshold. 
        The matching is the average of minimum RMSDs.
        Args : 
            ref_rdmol: Mol = RDKit molecule containing the reference conformations
            gen_rdmol: Mol = RDKit molecule containing the generated conformations
            rmsd_thresholds: list = thresholds for the coverage in angstrom
            remove_hydrogens : remove hydrogens to the input molecules, keeping only heavy atoms
        Returns :
            dict = coverages for each input threshold
            float = matching
        """
        
        if remove_hydrogens :
            ref_rdmol = Chem.RemoveHs(ref_rdmol)
            gen_rdmol = Chem.RemoveHs(gen_rdmol)
            
        assert Chem.MolToSmiles(ref_rdmol) == Chem.MolToSmiles(gen_rdmol)
        self.analyzed_molecule_pairs.append([ref_rdmol, gen_rdmol])
        
        ref_gen_rmsd_matrix = self._get_rmsd_matrix(ref_rdmol, gen_rdmol)
        
        ref_min_rmsd = ref_gen_rmsd_matrix.min(1)
        coverages = {} # {threshold : coverage}
        if rmsd_thresholds is None :
            rmsd_thresholds = self.rmsd_thresholds
        for threshold in rmsd_thresholds :
            coverages[threshold] = (ref_min_rmsd < threshold).mean()
        matching = ref_min_rmsd.mean()
        
        self.coverages.append(coverages)
        self.matchings.append(matching)
        
        return coverages, matching
    
    
    def plot_coverage_per_rotatable_bond_number(self, 
                                                threshold: int = 1
                                               ) -> None :
        """
        Analyze the coverage per rotatable bonds for analyzed molecules
        """
        
        assert len(self.coverages) > 0
        
        n_rot_bonds = []
        coverages = []
        for i, (ref_rdmol, gen_rdmol) in enumerate(self.analyzed_molecule_pairs) :
            n_rot_bonds.append(Chem.rdMolDescriptors.CalcNumRotatableBonds(ref_rdmol))
            coverages.append(self.coverages[i][threshold])
        
        coverage_1a = pd.DataFrame([[n, c] for n, c in zip(n_rot_bonds, coverages)])
        mean_coverages = coverage_1a.groupby(0).mean()
        mean_coverages.plot.line()
    
    
    def get_bond_wasserstein_distances(self,
                                ref_rdmol: Mol, 
                                gen_rdmol: Mol
                               ) -> np.ndarray :
        
        """Get the Wasserstein distances of bond distances among conformations 
        for each bond in the input molecule
        Args : 
            ref_rdmol: Mol = RDKit molecule containing the reference conformations
            gen_rdmol: Mol = RDKit molecule containing the generated conformations
        Returns :
            np.ndarray = Wassertein distances for each pair of correponding bonds
        """
        
        wass_dists = []
        
        bond_atom_ids = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in ref_rdmol.GetBonds()]
        for begin_atom_idx, end_atom_idx in bond_atom_ids :
            ref_bond_distances = self.analyser.get_conformers_interatom_distances(ref_rdmol, begin_atom_idx, end_atom_idx)
            gen_bond_distances = self.analyser.get_conformers_interatom_distances(gen_rdmol, begin_atom_idx, end_atom_idx)
            wass_dists.append(wasserstein_distance(ref_bond_distances, gen_bond_distances))
        
        wass_dists = np.array(wass_dists)
        
        return wass_dists
    
    
    def get_angles_wasserstein_distances(self,
                                ref_rdmol: Mol,
                                gen_rdmol: Mol
                               ) -> np.ndarray :
        
        """Get the Wasserstein distances of angles in degree among conformations 
        for each angle in the input molecule
        Args : 
            ref_rdmol: Mol = RDKit molecule containing the reference conformations
            gen_rdmol: Mol = RDKit molecule containing the generated conformations
        Returns :
            np.ndarray = Wassertein distances for each pair of correponding angles
        """
        
        wass_dists = []
        
        angles_atom_ids = [tuple(path) for path in FindAllPathsOfLengthN(ref_rdmol, 3, useBonds=False)]
        for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :
            ref_angles = self.analyser.get_conformers_angles_deg(ref_rdmol, begin_atom_idx, second_atom_idx, end_atom_idx)
            gen_angles = self.analyser.get_conformers_angles_deg(gen_rdmol, begin_atom_idx, second_atom_idx, end_atom_idx)
            wass_dists.append(wasserstein_distance(ref_angles, gen_angles))
        
        wass_dists = np.array(wass_dists)
        
        return wass_dists
    
    
    def get_dihedrals_wasserstein_distances(self,
                                ref_rdmol: Mol, 
                                gen_rdmol: Mol
                               ) -> np.ndarray :
        
        """Get the Wasserstein distances of dihedrals in degree among conformations 
        for each dihedral in the input molecule
        Args : 
            ref_rdmol: Mol = RDKit molecule containing the reference conformations
            gen_rdmol: Mol = RDKit molecule containing the generated conformations
        Returns :
            np.ndarray = Wassertein distances for each pair of correponding dihedrals
        """
        
        wass_dists = []
        
        dihedrals_atom_ids = [tuple(path) for path in FindAllPathsOfLengthN(ref_rdmol, 4, useBonds=False)]
        for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in dihedrals_atom_ids :
            first_dihedrals = self.analyser.get_conformers_dihedrals_deg(ref_rdmol, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx)
            second_dihedrals = self.analyser.get_conformers_dihedrals_deg(gen_rdmol, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx)
            wass_dists.append(wasserstein_distance(first_dihedrals, second_dihedrals))
        
        wass_dists = np.array(wass_dists)
        
        return wass_dists
    
    
#     def _get_rmsd_matrix(self,
#                          ref_rdmol: Mol, 
#                          gen_rdmol: Mol = None
#                         ) -> np.ndarray :
        
#         """ The function will return the RMSD matrices between conformations
#         from 2 different RDKit molecule (or the same molecule if gen_rdmol is None)
#         Args : 
#              ref_rdmol: Mol = RDKit molecule containing the reference conformations
#              gen_rdmol: Mol = RDKit molecule containing the generated conformations (optional)
#         Returns :
#             np.ndarray = matrix giving the pairwise RMSD between the reference (rows) and 
#                 the generated (cols) conformations (in the same order as input molecules)
#         """
        
#         ref_confs_ids = [conf.GetId() for conf in ref_rdmol.GetConformers()]
        
#         if gen_rdmol is None :
#             gen_rdmol = copy.deepcopy(ref_rdmol)
            
#         gen_confs_ids = [conf.GetId() for conf in gen_rdmol.GetConformers()]
                
#         rmsd_matrix = np.zeros((len(ref_confs_ids), len(ref_confs_ids))) if gen_rdmol is None else np.zeros((len(ref_confs_ids), len(gen_confs_ids)))    
            
#         ref_rdmols = [Chem.MolFromSmiles(Chem.MolToSmiles(ref_rdmol)) for i in ref_confs_ids]
#         for i in ref_confs_ids :
#             ref_rdmols[i].AddConformer(ref_rdmol.GetConformer(i))
        
#         gen_rdmols = [Chem.MolFromSmiles(Chem.MolToSmiles(gen_rdmol)) for i in gen_confs_ids]
#         for i in gen_confs_ids :
#             gen_rdmols[i].AddConformer(gen_rdmol.GetConformer(i))
            
#         for i, ref_conf in enumerate(ref_rdmols) :
#             for j, gen_conf in enumerate(gen_rdmols) :
#                 rmsd_matrix[i, j] = Chem.rdMolAlign.GetBestRMS(gen_conf, ref_conf) 
    
#         return rmsd_matrix
    
    
    # trying using decomposition above
    def _get_rmsd_matrix(self,
                         ref_rdmol: Mol, 
                         gen_rdmol: Mol = None
                        ) -> np.ndarray :
        
        """ The function will return the RMSD matrices between conformations
        from 2 different RDKit molecule (or the same molecule if gen_rdmol is None)
        Args : 
             ref_rdmol: Mol = RDKit molecule containing the reference conformations
             gen_rdmol: Mol = RDKit molecule containing the generated conformations (optional)
        Returns :
            np.ndarray = matrix giving the pairwise RMSD between the reference (rows) and 
                the generated (cols) conformations (in the same order as input molecules)
        """
        
        ref_confs_ids = [conf.GetId() for conf in ref_rdmol.GetConformers()]
        
        if gen_rdmol is None :
            gen_rdmol = copy.deepcopy(ref_rdmol)
            
        gen_confs_ids = [conf.GetId() for conf in gen_rdmol.GetConformers()]
                
        rmsd_matrix = np.zeros((len(ref_confs_ids), len(ref_confs_ids))) if gen_rdmol is None else np.zeros((len(ref_confs_ids), len(gen_confs_ids)))
            
        for i, ref_conf_id in enumerate(ref_confs_ids) :
            for j, gen_conf_id in enumerate(gen_confs_ids) :
                rmsd_matrix[i, j] = Chem.rdMolAlign.GetBestRMS(gen_rdmol, ref_rdmol, gen_conf_id, ref_conf_id, None, 10000)
    
        return rmsd_matrix
    
    
    
class ConfAnalyzer(object) :
    
    def __init__(self):
        torsion_file_name = 'TorLibv2.xml'
        if os.path.exists(torsion_file_name) :
            with open(torsion_file_name, 'r') as f :
                lines = f.readlines()
            self.torsion_patterns = [line.strip()[21:-2] for line in lines if 'smarts' in line]
        
        
    def get_conformers_interatom_distances(self,
                                           rdmol: Mol,
                                           begin_atom_idx: int,
                                           end_atom_idx: int
                                          ) -> np.ndarray :
        
        """ This function returns the bond distances between 2 atoms from each conformer
        in the given molecule
        Args : 
            rdmol: Mol = Input molecule containing conformers
            begin_atom_idx: int = Index of the first atom in bond
            end_atom_idx: int = Index of the second atom in bond
        Returns :
            np.ndarray : array of shape (n_confs_in_mol)
        """
        
        bond_distances = []
        for conf in rdmol.GetConformers() :
            bond_distances.append(GetBondLength(conf, begin_atom_idx, end_atom_idx))
            
        return np.array(bond_distances)
    
    
    def get_conformers_angles_deg(self,
                                  rdmol: Mol,
                                  begin_atom_idx: int,
                                  second_atom_idx: int,
                                  end_atom_idx: int
                                 ) -> np.ndarray :
        
        """ This function returns the angles between 3 atoms from each conformer
        in the given molecule. Angles are in Deg.
        TODO : add argument for returning radiants ?
        Args : 
            rdmol: Mol = Input molecule containing conformers
            begin_atom_idx: int = Index of the first atom in angle
            second_atom_idx: int = Index of the second atom in angle
            end_atom_idx: int = Index of the third atom in angle
        Returns :
            np.ndarray : array of shape (n_confs_in_mol)
        """
        
        angles = []
        for conf in rdmol.GetConformers() :
            angles.append(GetAngleDeg(conf, begin_atom_idx, second_atom_idx, end_atom_idx))
            
        return np.array(angles)
            
        
    def get_conformers_dihedrals_deg(self,
                                     rdmol: Mol,
                                     begin_atom_idx: int,
                                     second_atom_idx: int,
                                     third_atom_idx: int,
                                     end_atom_idx: int
                                    ) -> np.ndarray :
        
        """ This function returns the dihedrals between 4 atoms from each conformer
        in the given molecule. Dihedral angles are in Deg.
        TODO : add argument for returning radiants ?
        Args : 
            rdmol: Mol = Input molecule containing conformers
            begin_atom_idx: int = Index of the first atom in angle
            second_atom_idx: int = Index of the second atom in angle
            third_atom_idx: int = Index of the third atom in angle
            end_atom_idx: int = Index of the fourth atom in angle
        Returns :
            np.ndarray : array of shape (n_confs_in_mol)
        """
        
        dihedrals = []
        for conf in rdmol.GetConformers() :
            dihedrals.append(GetDihedralDeg(conf, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx))
            
        return np.array(dihedrals)
    
    
    def plot_distance_histograms(self, 
                                 bond_atom_ids: List[Tuple[int, int]], 
                                 mol_dict: Dict[str, Mol]
                                ) -> None :
    
        """
        Compute histograms of each bond (couple of atom ids) distances from different
        sets of conformations. 
        Args :
            bond_atom_ids: List[Tuple[int, int]] = contains the bonds represented by 
        (begin_atom_idx, end_atom_idx), corresponding idx in the input molecules
            mol_dict: Dict[str, Mol] = associate a set name to a molecule containing 
        corresponding conformations
        """
    
        for begin_atom_idx, end_atom_idx in bond_atom_ids :

            dists = []
            confset = []
            for mol_setname, mol in mol_dict.items() :
                mol_dists = self.get_conformers_interatom_distances(mol, begin_atom_idx, end_atom_idx)
                dists.append(mol_dists)
                confset = confset + [mol_setname] * len(mol_dists)

            ref_vs_gen_wasserstein = None
            if len(mol_dict) == 2 :
                ref_vs_gen_wasserstein = wasserstein_distance(dists[0], dists[1])
                
            dists = np.hstack(dists)

            dists_df = pd.DataFrame(list(zip(dists, confset)), columns=['Distance (A)', 'Conformation Set'])
            
            other_mol = copy.deepcopy(list(mol_dict.values())[0])
            Chem.rdDepictor.Compute2DCoords(other_mol)
            d2d = rdMolDraw2D.MolDraw2DCairo(350,300)
            d2d.drawOptions().addAtomIndices=True
            d2d.DrawMolecule(other_mol, highlightAtoms=[begin_atom_idx, end_atom_idx])
            d2d.FinishDrawing()
            d2d.WriteDrawingText('mol.png') 
            
            fig, ax = plt.subplots(figsize=(10, 5))
            title = f'{begin_atom_idx} - {end_atom_idx}'
            if ref_vs_gen_wasserstein is not None :
                title = title + f' : WD = {ref_vs_gen_wasserstein:0.3f}'
            plt.title(title)
            sns.histplot(data=dists_df, x='Distance (A)', hue='Conformation Set', bins=50, common_norm=False, stat='density')
            plt.ticklabel_format(style='plain')

            
            newax = fig.add_axes([0.5, 0.05, 1 ,1], anchor='NE')
            im = plt.imread('mol.png')
            newax.imshow(im)
            newax.axis('off')
            
            plt.show()
            plt.clf()
            
            
    def plot_angles_histograms(self, 
                               angles_atom_ids: List[Tuple[int, int, int]], 
                               mol_dict: Dict[str, Mol]
                              ) -> None :
        
        """
        Compute histograms of each angles (triplet of atom ids) in degrees from different
        sets of conformations.
        Args :
            angles_atom_ids: List[Tuple[int, int, int]] = contains the angles represented by 
        (begin_atom_idx, second_atom_idx, end_atom_idx), corresponding idx in the input molecules
            mol_dict: Dict[str, Mol] = associate a set name to a molecule containing 
        corresponding conformations
        """
        
        for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :

            angles = []
            confset = []
            for mol_setname, mol in mol_dict.items() :
                mol_angles = self.get_conformers_angles_deg(mol, begin_atom_idx, second_atom_idx, end_atom_idx)
                angles.append(mol_angles)
                confset = confset + [mol_setname] * len(mol_angles)
                
            angles = np.hstack(angles)

            other_mol = copy.deepcopy(list(mol_dict.values())[0])
            Chem.rdDepictor.Compute2DCoords(other_mol)
            d2d = rdMolDraw2D.MolDraw2DCairo(350,300)
            d2d.drawOptions().addAtomIndices=True
            d2d.DrawMolecule(other_mol, highlightAtoms=[begin_atom_idx, end_atom_idx])
            d2d.FinishDrawing()
            d2d.WriteDrawingText('mol.png') 
            
            angles_df = pd.DataFrame(list(zip(angles, confset)), columns=['Angle (Deg)', 'ConfSet'])

            fig, ax = plt.subplots(figsize=(10, 5))
            plt.title(f'{begin_atom_idx} - {second_atom_idx} - {end_atom_idx}')
            sns.histplot(data=angles_df, x='Angle (Deg)', hue='ConfSet', bins=50, common_norm=False, stat='density')
            
            newax = fig.add_axes([0.5, 0.05, 1 ,1], anchor='NE')
            im = plt.imread('mol.png')
            newax.imshow(im)
            newax.axis('off')
            
            plt.show()
            plt.clf()
            
            
    def plot_dihedrals_histograms(self, 
                                  dihedrals_atom_ids: List[Tuple[int, int, int, int]], 
                                  mol_dict: Dict[str, Mol]
                                 ) -> None :
        
        """
        Compute histograms of each dihedral angles (quadruplet of atom ids) in degrees from 
        different sets of conformations.
        Args :
            dihedrals_atom_ids: List[Tuple[int, int, int, int]] = contains the dihedral angles 
        represented by (begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx), 
        corresponding idx in the input molecules
            mol_dict: Dict[str, Mol] = associate a set name to a molecule containing 
        corresponding conformations
        """
        
        for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in dihedrals_atom_ids :

            dihedrals = []
            confsets = []
            for mol_setname, mol in mol_dict.items() :
                mol_dihedrals = self.get_conformers_dihedrals_deg(mol, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx)
                dihedrals.append(mol_dihedrals)
                confsets = confsets + [mol_setname] * len(mol_dihedrals)

            dihedrals = np.hstack(dihedrals)
            dihedrals_rad = np.deg2rad(dihedrals)
            coss = np.cos(dihedrals_rad)
            sins = np.sin(dihedrals_rad)
            
            unique_confsets = mol_dict.keys()
            for scale, confset in enumerate(unique_confsets) :
                scale = scale + 1 
                confset_index = [i
                                 for i, cs in enumerate(confsets)
                                 if confset == cs]
                coss[confset_index] = coss[confset_index] * scale
                sins[confset_index] = sins[confset_index] * scale
            
            dihedrals_df = pd.DataFrame({'Dihedral angle (Deg)' : dihedrals,
                                         'cos' : coss,
                                         'sin' : sins,
                                         'Conformation Set' : confsets})

            other_mol = copy.deepcopy(list(mol_dict.values())[0])
            Chem.rdDepictor.Compute2DCoords(other_mol)
            d2d = rdMolDraw2D.MolDraw2DCairo(350,300)
            d2d.drawOptions().addAtomIndices=True
            d2d.DrawMolecule(other_mol, highlightAtoms=[begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx])
            d2d.FinishDrawing()
            d2d.WriteDrawingText('mol.png') 

            # fig, ax = plt.subplots(figsize=(10, 5))
            # plt.title(f'{begin_atom_idx} - {second_atom_idx} - {third_atom_idx} - {end_atom_idx}')
            # #sns.histplot(data=dihedrals_df, x='Dihedral angle (Deg)', hue='ConfSet', binwidth=15, binrange=(-180, 180), common_norm=False, stat='density')
            # sns.histplot(data=dihedrals_df, 
            #              x='Dihedral angle (Deg)', 
            #              hue='Conformation Set', 
            #              binwidth=15, 
            #              binrange=(-180, 180), 
            #              multiple='stack')
            # plt.xlim(-180, 180)

            # newax = fig.add_axes([0.5, 0.05, 1 ,1], anchor='NE')
            # im = plt.imread('mol.png')
            # newax.imshow(im)
            # newax.axis('off')

            # plt.show()
            # plt.clf()
            
            fig, ax = plt.subplots(figsize=(5, 5))
            sns.scatterplot(data=dihedrals_df,
                            x='cos',
                            y='sin',
                            hue='Conformation Set')
            lims = (-len(unique_confsets) - 0.1, len(unique_confsets) + 0.1)
            plt.title(f'{begin_atom_idx} - {second_atom_idx} - {third_atom_idx} - {end_atom_idx}')
            plt.xlim(*lims)
            plt.ylim(*lims)
            plt.plot(0, 0, 'ko')
            newax = fig.add_axes([1, -0.05, 1 ,1], anchor='NE')
            im = plt.imread('mol.png')
            newax.imshow(im)
            newax.axis('off')
            plt.show()
            plt.clf()
            
    def plot_rotatable_histograms_ref_vs_gen(self,
                                             mol_dict: Dict[str, Mol]) :
        mol = list(mol_dict.values())[0]
        dihedrals_atom_ids = self.get_rotatable_bonds_atom_idx(mol)
        
#         for atom_ids in dihedrals_atom_ids :
#             bools = []
#             for pattern in self.torsion_patterns :
#                 mol_pattern = Chem.MolFromSmarts(pattern)
#                 if mol_pattern is not None :
#                     matches = mol.GetSubstructMatches(mol_pattern)
#                     for match in matches :
#                         bools.append(match == atom_ids)
#             print(any(bools))
        
        self.plot_dihedrals_histograms(dihedrals_atom_ids=dihedrals_atom_ids, mol_dict=mol_dict)
            
    def get_rotatable_bonds_atom_idx(self, mol) :
        rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
        dihedral_idxs = []
        for i, j in rot_atom_pairs :

            atom_i_bonds = iter(mol.GetAtomWithIdx(i).GetBonds())
            correct_bond = False
            while not correct_bond :
                next_bond = next(atom_i_bonds)
                begin, end = next_bond.GetBeginAtomIdx(), next_bond.GetEndAtomIdx()
                if i == begin :
                    if j != end :
                        rot_atom_indices = [end, begin]
                        correct_bond = True
                else :
                    if j != begin :
                        rot_atom_indices = [begin, end]
                        correct_bond = True

            atom_j_bonds = iter(mol.GetAtomWithIdx(j).GetBonds())
            correct_bond = False
            while not correct_bond :
                next_bond = next(atom_j_bonds)
                begin, end = next_bond.GetBeginAtomIdx(), next_bond.GetEndAtomIdx()
                if j == begin :
                    if i != end :
                        rot_atom_indices = rot_atom_indices + [begin, end]
                        correct_bond = True
                else :
                    if i != begin :
                        rot_atom_indices = rot_atom_indices + [end, begin]
                        correct_bond = True

            dihedral_idxs.append(tuple(rot_atom_indices))
            
        return(dihedral_idxs)
    
    
    def compare_dihedral_mol_to_library(self, 
                                    dihedral_atom_ids: Tuple[int, int, int, int],
                                    mol: Mol, 
                                    library: List[Mol]) :
        
        bonds = []
        begin, second, third, end = dihedral_atom_ids
        print(dihedral_atom_ids)
        bonds.append(mol.GetBondBetweenAtoms(begin, second).GetIdx())
        bonds.append(mol.GetBondBetweenAtoms(second, third).GetIdx())
        bonds.append(mol.GetBondBetweenAtoms(third, end).GetIdx())
        submol = Chem.PathToSubmol(mol, tuple(bonds))
        
        dihedrals = []
        for lib_mol in library :
            matches = lib_mol.GetSubstructMatches(submol)
            matches = self._get_correct_matches(matches, lib_mol)
            for match in matches :
                dihedrals.append(self.get_conformers_dihedrals_deg(lib_mol, *match))
               
        confset = ['Library'] * len(dihedrals)
            
        mol_dihedrals = self.get_conformers_dihedrals_deg(mol, *dihedral_atom_ids)
        dihedrals.append(mol_dihedrals)
        confset = confset + ['Molecule'] * len(mol_dihedrals)
        
        dihedrals = np.hstack(dihedrals)
        dihedrals_df = pd.DataFrame(list(zip(dihedrals, confset)), columns=['Dihedral angle (Deg)', 'ConfSet'])
        
        Chem.rdDepictor.Compute2DCoords(submol)
        d2d = rdMolDraw2D.MolDraw2DCairo(350,300)
        #d2d.drawOptions().addAtomIndices=True
        d2d.DrawMolecule(submol)
        d2d.FinishDrawing()
        d2d.WriteDrawingText('mol.png') 
        
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.title(f'{begin} - {second} - {third} - {end}')
        sns.histplot(data=dihedrals_df, x='Dihedral angle (Deg)', hue='ConfSet', binwidth=15, binrange=(-180, 180), common_norm=False, stat='density')
        plt.xlim(-180, 180)
        
        newax = fig.add_axes([0.5, 0.05, 1 ,1], anchor='NE')
        im = plt.imread('mol.png')
        newax.imshow(im)
        newax.axis('off')
        
        plt.show()
        plt.clf()
        
    def compare_libraries(self, 
                          library_dict: Dict[str, List[Mol]],
                         use_predefined_patterns: bool = True,
                         draw_patterns: bool = False) :
        
        if not use_predefined_patterns :
            dihedral_patterns = self._find_dihedral_patterns(library_dict)
            dihedral_patterns = [smarts for smarts, count in dihedral_patterns.most_common()]
        else :
            dihedral_patterns = self.torsion_patterns
        
        dihedral_patterns = dihedral_patterns[:10]
        
        for smarts in dihedral_patterns :
            pattern = Chem.MolFromSmarts(smarts)
            #Chem.SanitizeMol(pattern) # it making the predefined patterns bug
            dihedrals = []
            confset = []
            for library_name, mol_list in library_dict.items() :
                lib_dihedrals = []
                for mol in mol_list :
                    matches = mol.GetSubstructMatches(pattern)
                    matches = self._get_correct_matches(matches, mol)
                    for match in matches :
                        lib_dihedrals.append(self.get_conformers_dihedrals_deg(mol, *match))
                confset = confset + [library_name] * len(lib_dihedrals)
                dihedrals = dihedrals + lib_dihedrals
        
            if len(dihedrals) :
                dihedrals = np.hstack(dihedrals)
                dihedrals_df = pd.DataFrame(list(zip(dihedrals, confset)), columns=['Dihedral angle (Deg)', 'ConfSet'])

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.title(smarts)
                sns.histplot(data=dihedrals_df, x='Dihedral angle (Deg)', hue='ConfSet', binwidth=15, binrange=(-180, 180), common_norm=False, stat='density')
                plt.xlim(-180, 180)

                if draw_patterns :
                    smarts = urllib.parse.quote(smarts)
                    im = mpimg.imread("https://smarts.plus/smartsview/download_rest?smarts="+smarts,format="png")
                    newax = fig.add_axes([0.5, 0.05, 1 ,1], anchor='NE')
                    newax.imshow(im)
                    newax.axis('off')

                #plt.savefig(smarts, bbox_inches='tight')
                plt.show()
                plt.clf()
        
    def _get_correct_matches(self, matches, mol) : # to get the right atom order
        correct_matches = []
        dihedrals_atom_ids = [tuple(path) for path in FindAllPathsOfLengthN(mol, 4, useBonds=False)]
        for match in matches :
            for dihedral in dihedrals_atom_ids :
                if sorted(dihedral) == sorted(match) :
                    correct_matches.append(dihedral)

        return correct_matches
    
    def _find_dihedral_patterns(self, library_dict) :
        dihedral_patterns = Counter()
        for library_name, mol_list in library_dict.items() :
            for mol in mol_list :
                dihedrals_bond_ids = [tuple(path) for path in FindAllPathsOfLengthN(mol, 3, useBonds=True)]
                for dihedral_bond_ids in dihedrals_bond_ids :
                    submol = Chem.PathToSubmol(mol, dihedral_bond_ids)
                    smarts = Chem.MolToSmarts(submol)
                    dihedral_patterns.update([smarts])
        return dihedral_patterns