import os
import numpy as np
import pickle
import logging

from typing import Union
from tqdm import tqdm
from collections import defaultdict
from rdkit.Chem import Mol
from genbench3d.utils import shift_torsion_values
from .geometry_extractor import GeometryExtractor
from .pattern import BondPattern, AnglePattern, TorsionPattern, GeometryPattern
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from .von_mises_kde import VonMisesKDE
from scipy.stats import norm
from dataclasses import dataclass
from genbench3d.data.source import DataSource
from scipy.optimize import minimize, OptimizeResult
from scipy.special import iv

VALIDITY_METHODS = ['boundaries', 'mixtures', 'kernel_density']

@dataclass
class GeometryMixture:
    mixture: GaussianMixture
    max_likelihood: float
    shift: float = 0 # only used in torsion mixtures
    
@dataclass
class GeometryKernelDensity:
    kernel_density: KernelDensity
    max_likelihood: float
    
Range = tuple[float, float]
    
Values = dict[GeometryPattern, list[float]]
Ranges = dict[GeometryPattern, Range]
Mixtures = dict[GeometryPattern, GeometryMixture]
KernelDensities = dict[GeometryPattern, GeometryKernelDensity]

class ReferenceGeometry():
    
    def __init__(self,
                 source: DataSource,
                 root: str,
                 min_pattern_values: int,
                 validity_method: str = 'kernel_density',
                 ) -> None:
        assert validity_method in VALIDITY_METHODS, \
            f'Validity method must be in {VALIDITY_METHODS}'
            
        self.source = source
        self.root = root
        self.validity_method = validity_method
        self.min_pattern_values = min_pattern_values
        
        self.geometry_extractor = GeometryExtractor()
        
        self.values_filepath = os.path.join(root, f'{source.name}_geometry_values.p')
        self.ranges_filepath = os.path.join(root, f'{source.name}_geometry_ranges.p')
        self.mixtures_filepath = os.path.join(root, f'{source.name}_geometry_mixtures.p')
        self.kernel_densities_filepath = os.path.join(root, f'{source.name}_geometry_kernel_densities.p')
        
        if validity_method == 'boundaries':
            self.ranges = self.read_ranges()
        if validity_method == 'mixtures':
            self.mixtures = self.read_mixtures()
        if validity_method == 'kernel_density':
            self.kernel_densities = self.read_densities()
    
    
    def read_values(self) -> dict[str, Values]:
        if not os.path.exists(self.values_filepath):
            values = self.compute_values()
        else:
            with open(self.values_filepath, 'rb') as f:
                values = pickle.load(f)
        return values
    
    
    def read_ranges(self) -> dict[str, Ranges]:
        values = self.read_values()
        if not os.path.exists(self.ranges_filepath):
            ranges = self.compute_ranges(values)
        else:
            with open(self.ranges_filepath, 'rb') as f:
                ranges = pickle.load(f)
        return ranges
    
    
    def read_mixtures(self) -> dict[str, Mixtures]:
        values = self.read_values()
        if not os.path.exists(self.mixtures_filepath):
            mixtures = self.compute_mixtures(values)
        else:
            with open(self.mixtures_filepath, 'rb') as f:
                mixtures = pickle.load(f)
        return mixtures
    
    
    def read_densities(self) -> dict[str, KernelDensities]:
        values = self.read_values()
        if not os.path.exists(self.kernel_densities_filepath):
            kernel_densities = self.compute_densities(values)
        else:
            with open(self.kernel_densities_filepath, 'rb') as f:
                kernel_densities = pickle.load(f)
        return kernel_densities
    
    
    def compute_values(self) -> dict[str, Values]:
        
        logging.info(f'Compiling geometry values for {self.source.name}')
        
        all_bond_values = defaultdict(list)
        all_angle_values = defaultdict(list)
        all_torsion_values = defaultdict(list)
        
        # for mol in tqdm(mol_iterator):
        for mol in tqdm(self.source):
            if mol is not None:
                assert isinstance(mol, Mol)
                
                mol_bond_values = self.get_mol_bond_lengths(mol)
                for bond_pattern, bond_values in mol_bond_values.items():
                    all_bond_values[bond_pattern].extend(bond_values)
                
                mol_angle_values = self.get_mol_angle_values(mol)
                for angle_pattern, angle_values in mol_angle_values.items():
                    all_angle_values[angle_pattern].extend(angle_values)
                    
                mol_torsion_values = self.get_mol_torsion_values(mol)
                for torsion_pattern, torsion_values in mol_torsion_values.items():
                    all_torsion_values[torsion_pattern].extend(torsion_values)
        
        values = {'bond': all_bond_values,
                  'angle': all_angle_values,
                  'torsion': all_torsion_values}
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
            
        return values
    
    
    def compute_ranges(self,
                       values: dict[str, Values]
                       ) -> Ranges:
        
        logging.info(f'Computing geometry ranges for {self.source.name}')
        
        ranges = self.get_ranges_from_values(bond_values=values['bond'],
                                             angle_values=values['angle'],
                                             torsion_values=values['torsion'])
        
        with open(self.ranges_filepath, 'wb') as f:
            pickle.dump(ranges, f)
            
            
    def compute_mixtures(self,
                         values: dict[str, Values]
                         ) -> dict[str, Mixtures]:
        
        logging.info(f'Computing geometry mixtures for {self.source.name}')
        
        bond_values: dict[BondPattern, list[float]] = values['bond']
        angle_values: dict[AnglePattern, list[float]] = values['angle']
        torsion_values: dict[TorsionPattern, list[float]] = values['torsion']
        
        # Bonds
        logging.info(f'Computing bond mixtures for {self.source.name}')
        bond_mixtures: dict[BondPattern, GeometryMixture] = {}
        for pattern, values in tqdm(bond_values.items()):
            if len(values) > self.min_pattern_values:
                mixture = self.get_mixture(values)
                max_likelihood = self.get_max_likelihood(mixture, values, geometry='bond')
                geometry_mixture = GeometryMixture(mixture, max_likelihood)
                bond_mixtures[pattern] = geometry_mixture
                
        logging.info(f'Computing mixtures for generalized bond patterns for {self.source.name}')
        generalized_bond_values = defaultdict(list)
        for pattern, values in bond_values.items():
            generalized_pattern = pattern.generalize()
            generalized_bond_values[generalized_pattern].extend(values)
            
        for pattern, values in tqdm(generalized_bond_values.items()):
            if len(values) > self.min_pattern_values:
                mixture = self.get_mixture(values)
                max_likelihood = self.get_max_likelihood(mixture, values, geometry='bond')
                geometry_mixture = GeometryMixture(mixture, max_likelihood)
                bond_mixtures[pattern] = geometry_mixture
                
        # Angles
        logging.info(f'Computing angle mixtures for {self.source.name}')
        angle_mixtures: dict[AnglePattern, GeometryMixture] = {}
        for pattern, values in tqdm(angle_values.items()):
            if len(values) > self.min_pattern_values:
                mixture = self.get_mixture(values)
                max_likelihood = self.get_max_likelihood(mixture, values, geometry='angle')
                geometry_mixture = GeometryMixture(mixture, max_likelihood)
                angle_mixtures[pattern] = geometry_mixture
                
        # Generalized angle pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized angle mixtures for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_angle_values = defaultdict(list)
            for pattern, values in angle_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_angle_values[generalized_pattern].extend(values)
                
            for pattern, values in tqdm(generalized_angle_values.items()):
                if len(values) > 50:
                    mixture = self.get_mixture(values)
                    max_likelihood = self.get_max_likelihood(mixture, values, geometry='angle')
                    geometry_mixture = GeometryMixture(mixture, max_likelihood)
                    angle_mixtures[pattern] = geometry_mixture
                
        # Torsions
        def get_torsion_mixture(values):
            abs_values = np.abs(values)
            samples = np.linspace(0, 180, 181).reshape(-1, 1)
            distances = np.abs(samples - abs_values)
            min_distances = np.min(distances, axis=1)
            
            # the shift is the point further away from all values
            # = the "minimum" on the torsion space
            max_i = np.argmax(min_distances) 
            shifted_values = shift_torsion_values(values=values,
                                                        x=max_i)
            mixture = self.get_mixture(shifted_values)
            max_likelihood = self.get_max_likelihood(mixture, shifted_values, geometry='torsion')
            torsion_mixture = GeometryMixture(mixture, max_likelihood, max_i)
            return torsion_mixture
            
        logging.info(f'Computing torsion mixtures for {self.source.name}')
        torsion_mixtures = {}
        for pattern, values in tqdm(torsion_values.items()):
            if len(values) > 50:
                torsion_mixture = get_torsion_mixture(values)
                torsion_mixtures[pattern] = torsion_mixture
                
        # Generalized torsion pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized torsion mixtures for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_torsion_values = defaultdict(list)
            for pattern, values in torsion_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_torsion_values[generalized_pattern].extend(values)
                
            for pattern, values in tqdm(generalized_torsion_values.items()):
                if len(values) > 50:
                    torsion_mixture = get_torsion_mixture(values)
                    torsion_mixtures[pattern] = torsion_mixture
                
        mixtures = {'bond': bond_mixtures,
                    'angle': angle_mixtures,
                    'torsion': torsion_mixtures}
        
        with open(self.mixtures_filepath, 'wb') as f:
            pickle.dump(mixtures, f)
            
        return mixtures
            
            
    def compute_densities(self,
                          values: dict[str, Values]
                         ) -> dict[str, Mixtures]:
        
        logging.info(f'Computing geometry kernel densities for {self.source.name}')
        
        bond_values: dict[BondPattern, list[float]] = values['bond']
        angle_values: dict[AnglePattern, list[float]] = values['angle']
        torsion_values: dict[TorsionPattern, list[float]] = values['torsion']
        
        bond_bandwidth = 0.01
        angle_bandwidth = 1.0
        torsion_bandwidth = 200.0
        
        # Bonds
        logging.info(f'Computing bond kernel densities for {self.source.name}')
        bond_kernel_densities: dict[BondPattern, GeometryKernelDensity] = {}
        for pattern, pattern_values in tqdm(bond_values.items()):
            if len(pattern_values) > self.min_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 10
                bandwidth = bond_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         values=pattern_values, 
                                                         geometry='bond')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood)
                bond_kernel_densities[pattern] = geometry_kernel_density
                
        logging.info(f'Computing kernel densities for generalized bond patterns for {self.source.name}')
        generalized_bond_values = defaultdict(list)
        for pattern, values in bond_values.items():
            generalized_pattern = pattern.generalize()
            generalized_bond_values[generalized_pattern].extend(values)
            
        for pattern, pattern_values in tqdm(generalized_bond_values.items()):
            if len(pattern_values) > self.min_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 10
                bandwidth = bond_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         pattern_values, 
                                                         geometry='bond')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood)
                bond_kernel_densities[pattern] = geometry_kernel_density
                
        # Angles
        logging.info(f'Computing angle kernel densities for {self.source.name}')
        angle_kernel_densities: dict[AnglePattern, GeometryKernelDensity] = {}
        for pattern, pattern_values in tqdm(angle_values.items()):
            if len(pattern_values) > self.min_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 5
                bandwidth = angle_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         pattern_values, 
                                                         geometry='angle')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood)
                angle_kernel_densities[pattern] = geometry_kernel_density
                
        # Generalized angle pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized angle kernel densities for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_angle_values = defaultdict(list)
            for pattern, pattern_values in angle_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_angle_values[generalized_pattern].extend(pattern_values)
                
            for pattern, pattern_values in tqdm(generalized_angle_values.items()):
                if len(pattern_values) > 50:
                    # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                    # bandwidth = bandwidth * 5
                    bandwidth = angle_bandwidth
                    kernel_density = KernelDensity(bandwidth=bandwidth)
                    kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                    max_likelihood = self.get_max_likelihood(kernel_density, 
                                                            pattern_values, 
                                                            geometry='angle')
                    geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                    max_likelihood)
                    angle_kernel_densities[pattern] = geometry_kernel_density
            
        logging.info(f'Computing torsion kernel densities for {self.source.name}')
        torsion_kernel_densities = {}
        for pattern, pattern_values in tqdm(torsion_values.items()):
            if len(pattern_values) > 50:
                torsion_rad = np.radians(pattern_values)
                # bandwidth = self.taylor_von_mises_bandwidth_estimation(torsion_rad)
                bandwidth = torsion_bandwidth
                kernel_density = VonMisesKDE(bandwidth=bandwidth)
                kernel_density.fit(np.array(torsion_rad).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                        torsion_rad, 
                                                        geometry='torsion')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood)
                torsion_kernel_densities[pattern] = geometry_kernel_density
                
        # Generalized torsion pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized torsion kernel densities for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_torsion_values = defaultdict(list)
            for pattern, pattern_values in torsion_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_torsion_values[generalized_pattern].extend(pattern_values)
                
            for pattern, pattern_values in tqdm(generalized_torsion_values.items()):
                if len(pattern_values) > 50:
                    torsion_rad = np.radians(pattern_values)
                    # bandwidth = self.taylor_von_mises_bandwidth_estimation(torsion_rad)
                    bandwidth = torsion_bandwidth
                    kernel_density = VonMisesKDE(bandwidth=bandwidth)
                    kernel_density.fit(np.array(torsion_rad).reshape(-1, 1))
                    max_likelihood = self.get_max_likelihood(kernel_density, 
                                                            torsion_rad, 
                                                            geometry='torsion')
                    geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                    max_likelihood)
                    torsion_kernel_densities[pattern] = geometry_kernel_density
                
        kernel_densities = {'bond': bond_kernel_densities,
                            'angle': angle_kernel_densities,
                            'torsion': torsion_kernel_densities}
        
        with open(self.kernel_densities_filepath, 'wb') as f:
            pickle.dump(kernel_densities, f)
            
        return kernel_densities
            
            
    def silverman_scott_bandwidth_estimation(self, 
                                             values: list[float]) -> float:
        n = len(values)
        std = np.std(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        bandwidth = 1.06 * np.min([std, iqr / 1.34]) * n ** (-1/5)
        return bandwidth
    
    
    def taylor_von_mises_bandwidth_estimation(self,
                                                values: list[float],
                                                bandwidth_min_value: float = 10.0,
                                                bandwidth_max_value: float = 300.0,
                                                kappa_max_value: float = 200.0) -> float:
        n = len(values)
        C = np.sum(np.cos(values))
        S = np.sum(np.sin(values))
        R_dash = np.sqrt((C ** 2) + (S ** 2)) / n
        if R_dash < 0.53:
            kappa = 2 * R_dash + (R_dash ** 3) + 5 * (R_dash ** 5) / 6
        elif R_dash < 0.85:
            kappa = -0.4 + 1.39 * R_dash + 0.43 / (1 - R_dash)
        elif R_dash < 1.0:
            kappa = 1 / ((R_dash ** 3) - 4 * (R_dash ** 2) + 3 * R_dash)
            if kappa > kappa_max_value:
                return bandwidth_max_value
        else:
            return bandwidth_max_value
        
        # kappa = np.min([kappa, 100]) # high kappa values leads to iv(0, kappa) = np.inf by overflow
        
        num = 3 * n * np.square(kappa) * iv(2, 2 * kappa)
        den = (4 * np.power(np.pi, 1/2) * np.square(iv(0, kappa)))
        # if kappa > 1000000 or den == np.inf:
        #     import pdb;pdb.set_trace()
        bandwidth = np.power(num / den, 2 / 5)
        bandwidth = np.min([bandwidth, bandwidth_max_value])
        bandwidth = np.max([bandwidth, bandwidth_min_value])
        
        return bandwidth
            
    def get_mixture(self,
                    values: list[float]) -> GaussianMixture:
        values = np.array(values).reshape(-1, 1)
        best_gm = GaussianMixture(1)
        best_gm.fit(values)
        min_bic = best_gm.bic(values)
        for n_components in range(2, 7):
            gm = GaussianMixture(n_components)
            gm.fit(values)
            bic = gm.bic(values)
            if bic < min_bic:
                best_gm = gm
                min_bic = bic
        return best_gm
        
        
    def get_likelihoods(self,
                       gaussian_mixture: GaussianMixture,
                       values: list[float]) -> list[float]:
        means = gaussian_mixture.means_.reshape(-1)
        stds = np.sqrt(gaussian_mixture.covariances_).reshape(-1)
        weights = gaussian_mixture.weights_.reshape(-1)
        likelihoods = []
        for mean, std, weight in zip(means, stds, weights):
            g_likelihoods = norm.pdf(values, loc=mean, scale=std)
            max_likelihood = norm.pdf(mean, loc=mean, scale=std)
            norm_g_likelihoods = g_likelihoods / max_likelihood
            likelihoods.append(norm_g_likelihoods * weight)
            
        return np.sum(likelihoods, axis=0)
        
        
    def get_max_likelihood(self,
                           estimator: Union[GaussianMixture, KernelDensity],
                           values: list[float],
                           geometry: str,
                           n_samples: int = 1000,
                           ) -> float:
        
        def get_neg_log_likelihood(value: np.ndarray):
            log_likelihood = estimator.score_samples(value.reshape(-1, 1))
            return -log_likelihood
        
        # values = mixture.means_.reshape(-1)
        if geometry == 'bond':
            samples = np.linspace(0.5, 3.5, n_samples)
        elif geometry == 'angle':
            samples = np.linspace(0, 180, n_samples)
        else:
            if isinstance(estimator, GaussianMixture):
                samples = np.linspace(-180, 180, n_samples)
            else:
                assert isinstance(estimator, KernelDensity) or isinstance(estimator, VonMisesKDE)
                samples = np.linspace(-np.pi, np.pi, n_samples)
        # if isinstance(estimator, GaussianMixture):
        #     likelihoods = self.get_likelihoods(estimator, samples)
        #     max_likelihood = np.max(likelihoods)
        # else:
        #     assert isinstance(estimator, KernelDensity) or isinstance(estimator, VonMisesKDE)
        log_likelihoods = estimator.score_samples(np.array(samples).reshape(-1, 1))
        likelihood_argmax = np.argmax(log_likelihoods)
        max_likelihood_sample = samples[likelihood_argmax]
        
        if geometry == 'bond':
            bounds = [(np.min(values), np.max(values))]
        elif geometry == 'angle':
            bounds = [(np.min(values), 180)]
        else:
            bounds = [(-np.pi, np.pi)]
        
        result: OptimizeResult = minimize(fun=get_neg_log_likelihood, 
                                            x0=max_likelihood_sample, 
                                            method='Nelder-Mead',
                                            bounds=bounds)
        min_neg_log_likelihood = result.fun
        max_log_likelihood = -min_neg_log_likelihood
        max_likelihood = np.exp(max_log_likelihood)
        
        # log_likelihoods = estimator.score_samples(np.array(values).reshape(-1, 1))
        # likelihood_argmax = np.argmax(log_likelihoods)
        # max_likelihood = np.exp(log_likelihoods[likelihood_argmax])
        return max_likelihood
            
            
    def get_ranges_from_values(self,
                               bond_values: Values,
                                angle_values: Values,
                                torsion_values: Values):
        
        bond_ranges = {}
        for bond_pattern, values in bond_values.items():
            ranges = self.compute_authorized_ranges_bond(values)
            bond_ranges[bond_pattern] = ranges
        
        angle_ranges = {}
        for angle_pattern, values in angle_values.items():
            ranges = self.compute_authorized_ranges_angle(values)
            angle_ranges[angle_pattern] = ranges
            
        torsion_ranges = {}
        for torsion_pattern, values in torsion_values.items():
            ranges = self.compute_authorized_ranges_torsion(values)
            torsion_ranges[torsion_pattern] = ranges
        
        ranges = {
            'bond': bond_ranges,
            'angle': angle_ranges,
            'torsion': torsion_ranges,
            }
        
        return ranges
              
              
    def get_mol_bond_lengths(self,
                             mol: Mol) -> Values:
        bond_values = defaultdict(list)
        for bond in mol.GetBonds():
            bond_pattern = self.geometry_extractor.get_bond_pattern(bond)
            for conf in mol.GetConformers():
                bond_length = self.geometry_extractor.get_bond_length(conf, bond)
                bond_values[bond_pattern].append(bond_length)
        return bond_values
    
    
    def get_mol_angle_values(self,
                             mol: Mol) -> Values:
        angle_values = defaultdict(list)
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :
            
            angle_pattern = self.geometry_extractor.get_angle_pattern(mol, 
                                                                    begin_atom_idx, 
                                                                    second_atom_idx, 
                                                                    end_atom_idx)
                
            for conf in mol.GetConformers():
                angle_value = self.geometry_extractor.get_angle_value(conf,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        end_atom_idx)
                angle_values[angle_pattern].append(angle_value)
                
        return angle_values
        
                    
    def get_mol_torsion_values(self,
                               mol: Mol,
                               ) -> Values:
        
        torsion_values = defaultdict(list)
        torsion_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsion_atom_ids :
            
            torsion_pattern = self.geometry_extractor.get_torsion_pattern(mol,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
            for conf in mol.GetConformers():
                torsion_value = self.geometry_extractor.get_torsion_value(conf,
                                                                            begin_atom_idx,
                                                                            second_atom_idx,
                                                                            third_atom_idx,
                                                                            end_atom_idx)
                if not np.isnan(torsion_value):
                    torsion_values[torsion_pattern].append(torsion_value)
                        
        return torsion_values
    
    
    def compute_authorized_ranges_bond(self,
                                       values, 
                                       authorized_distance: float = 0.025) -> list[Range]:
        min_value = np.around(np.min(values) - authorized_distance, 3)
        max_value = np.around(np.max(values) + authorized_distance, 3)
        return [(min_value, max_value)]
    
    
    def compute_authorized_ranges_angle(self,
                                        values, 
                                        nbins: int = 36, 
                                        binrange: list[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5 # Deg
                                        ) -> list[Range]:
        if len(values) > self.min_pattern_values:
            counts, binticks = np.histogram(values, 
                                            bins=nbins, 
                                            range=binrange)
            authorized_ranges = []
            in_range = False # cursor to indicate whether we are inside possible value range
            # We scan all bins of the histogram, and merge bins accordinly to create ranges
            for i, (value, tick) in enumerate(zip(counts, binticks)) :
                if value > 0:
                    if not in_range: # if we have values and we were not in range, we enter in range
                        start = tick
                        in_range = True
                else:
                    if in_range: # if we have no values and we were in range, we exit range
                        end = tick
                        authorized_ranges.append((start, end))
                        in_range = False
            if in_range:
                end = binticks[-1]
                authorized_ranges.append((start, end))

            new_ranges = []
            for start, end in authorized_ranges:
                new_start = max(start - authorized_distance, binrange[0])
                new_end = min(end + authorized_distance, binrange[1])
                new_range = (new_start, new_end)
                new_ranges.append(new_range)

            corrected_ranges = []
            previous_range = new_ranges[0]
            for new_range in new_ranges[1:]:
                if new_range[0] <= previous_range[1]:
                    previous_range = (previous_range[0], new_range[1])
                else:
                    corrected_ranges.append(previous_range)
                    previous_range = new_range
            corrected_ranges.append(previous_range)

            return corrected_ranges
        else:
            return [tuple(binrange)]
    
    
    def compute_authorized_ranges_torsion(self,
                                          values, 
                                        nbins: int = 36, 
                                        binrange: list[float] = [0.0, 180.0], 
                                        authorized_distance: float = 2.5,
                                        absolute_torsion: bool = True) -> list[Range]:
        if absolute_torsion:
            values = np.abs(values)
        return self.compute_authorized_ranges_angle(values, 
                                                    nbins,
                                                    binrange, 
                                                    authorized_distance)
        
    
    def geometry_is_valid(self, 
                          geometry_pattern: GeometryPattern, 
                          value: float, 
                          geometry: str = 'bond',
                          new_pattern_is_valid: bool = True,
                          ) -> tuple[float, bool]:
        
        assert geometry in ['bond', 'angle', 'torsion']
        
        if self.validity_method == 'kernel_density':
            if geometry == 'bond':
                kds = self.kernel_densities['bond']
            elif geometry == 'angle':
                kds = self.kernel_densities['angle']
            elif geometry == 'torsion':
                kds = self.kernel_densities['torsion']
                value = np.radians(value)
            else:
                raise RuntimeError()
            
            q_value = np.nan
            new_pattern = False
            if geometry_pattern in kds:
                kd = kds[geometry_pattern]
                kernel_density = kd.kernel_density
                log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
                likelihood = np.exp(log_likelihood)
                q_value = likelihood.item() / kd.max_likelihood
                
            else:
                generalized_pattern = geometry_pattern.generalize() 
                logging.debug(f'Trying to generalize pattern (outer) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
                if generalized_pattern in kds:
                    kd = kds[generalized_pattern]
                    kernel_density = kd.kernel_density
                    log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
                    likelihood = np.exp(log_likelihood)
                    q_value = likelihood.item() / kd.max_likelihood
                else:
                    if geometry == 'bond':
                        new_pattern = True
                    else:
                        generalized_pattern = geometry_pattern.generalize(inner_neighbors=True)
                        logging.debug(f'Trying to generalize pattern (outer + inner) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
                        if generalized_pattern in kds:
                            kd = kds[generalized_pattern]
                            kernel_density = kd.kernel_density
                            log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
                            likelihood = np.exp(log_likelihood)
                            q_value = likelihood.item() / kd.max_likelihood
                        else:
                            new_pattern = True
        
        elif self.validity_method == 'boundaries':
        
            if geometry == 'bond':
                ranges = self.ranges['bond']
            elif geometry == 'angle' :
                ranges = self.ranges['angle']
            elif geometry == 'torsion':
                ranges = self.ranges['torsion']
                value = np.abs(value) # this method is done on absolute torsion ranges [0, 180]
            else:
                raise RuntimeError()
            
            in_range = False
            new_pattern = False
            if geometry_pattern in ranges:
                current_ranges = ranges[geometry_pattern]

                for range in current_ranges:
                    if value >= range[0] and value <= range[1]:
                        in_range = True
                        break
                    
            else:
                new_pattern = True
                if new_pattern_is_valid:
                    in_range = True
                
            q_value = float(in_range)
        
        elif self.validity_method == 'mixtures':
            
            if geometry == 'bond':
                mixtures = self.mixtures['bond']
            elif geometry == 'angle' :
                mixtures = self.mixtures['angle']
            elif geometry == 'torsion':
                mixtures = self.mixtures['torsion']
                
            else:
                raise RuntimeError()
            
            q_value = np.nan
            new_pattern = False
            if geometry_pattern in mixtures:
                geometry_mixture = mixtures[geometry_pattern]
                q_value = self.get_q_value(value, 
                                           geometry_pattern, 
                                           geometry_mixture,
                                           geometry)
                    
            else:
                # no args: default only outer neighbors are removed (there is no inner in bonds)
                generalized_pattern = geometry_pattern.generalize() 
                logging.debug(f'Trying to generalize pattern (outer) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
                if generalized_pattern in mixtures:
                    geometry_mixture = mixtures[generalized_pattern]
                    q_value = self.get_q_value(value, 
                                                geometry_pattern, 
                                                geometry_mixture,
                                                geometry)
                else:
                    if geometry == 'bond':
                        new_pattern = True
                    else:
                        generalized_pattern = geometry_pattern.generalize(inner_neighbors=True)
                        logging.debug(f'Trying to generalize pattern (outer + inner) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
                        if generalized_pattern in mixtures:
                            geometry_mixture = mixtures[generalized_pattern]
                            q_value = self.get_q_value(value, 
                                                        geometry_pattern, 
                                                        geometry_mixture,
                                                        geometry)
                        else:
                            new_pattern = True
                
            if q_value > 1.1:
                # q-value might go slighly over 1 because of precision
                # But should not go over 1.1
                logging.warning(f'q-value = {q_value} > 1.1 for pattern {geometry_pattern.to_string()} with value {value}')
                
            if new_pattern:
                logging.debug(f'New pattern : {generalized_pattern.to_string()}')
                
            q_value = np.min([1, q_value]) # we might not have sampled the best likelihood
                
        return q_value, new_pattern
        
    
    def get_q_value(self,
                    value: float,
                    geometry_pattern: GeometryPattern,
                    geometry_mixture: GeometryMixture,
                    geometry: str
                    ) -> float:
        mixture = geometry_mixture.mixture
        max_likelihood = geometry_mixture.max_likelihood
        if geometry == 'torsion' :
            shift = geometry_mixture.shift
            bond_type_23 = geometry_pattern.bond_type_23
            if bond_type_23 == 3:
                # triple bond can have very variable torsion angles
                # and angles should be checked anyway
                q_value = 1
            else:
                values = [value]
                values = shift_torsion_values(values=values, x=shift)
                likelihood = self.get_likelihoods(mixture, values=values)
                q_value = likelihood.item() / max_likelihood
        
        else: # bond, angle
            values = [value]
            likelihood = self.get_likelihoods(mixture, values=values)
            q_value = likelihood.item() / max_likelihood
            
        return q_value