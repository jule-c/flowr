import logging
import os
from typing import Any, Dict, List

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from genbench3d.geometry import ReferenceGeometry

from .conf_ensemble import ConfEnsembleLibrary, GeneratedCEL
from .data.generated_sample_reader import ASEDBReader
from .metrics import (
    Diversity2D,
    Diversity3D,
    MaxTrainSim,
    Metric,
    MolProperty,
    Novelty2D,
    Novelty3D,
    RingProportion,
    StrainEnergy,
    TrainingMetric,
    Uniqueness2D,
    Uniqueness3D,
    Validity2D,
    Validity3D,
)


class GenBench3D:

    def __init__(
        self,
        reference_geometry: ReferenceGeometry,
        config: dict,
    ) -> None:

        self.reference_geometry = reference_geometry
        self.tfd_threshold = config["tfd_threshold"]
        self.q_value_threshold = config["q_value_threshold"]
        self.steric_clash_safety_ratio = config["steric_clash_safety_ratio"]
        self.maximum_ring_plane_distance = config["maximum_ring_plane_distance"]
        self.include_torsions_in_validity3D = config["include_torsions_in_validity3D"]
        self.consider_hydrogens = config["consider_hydrogens"]

        self.training_cel = None

    def set_training_mols(self, training_mols: List[Mol]) -> None:
        self.training_cel = ConfEnsembleLibrary.from_mol_list(training_mols)

    def set_training_cel(self, training_cel: ConfEnsembleLibrary):
        self.training_cel = training_cel

    def initialize(self) -> None:

        self.graph_metrics: List[Metric] = [
            Validity2D(),
            Uniqueness2D(),
            Diversity2D(),
            RingProportion(),
        ]
        self.mol_properties: List[str] = ["MW", "logP", "SAScore", "QED"]
        for property_name in self.mol_properties:
            self.graph_metrics.append(MolProperty(name=property_name))

        self.training_graph_metrics: List[TrainingMetric] = [Novelty2D(), MaxTrainSim()]

        self.validity3D = Validity3D(
            reference_geometry=self.reference_geometry,
            q_value_threshold=self.q_value_threshold,
            steric_clash_safety_ratio=self.steric_clash_safety_ratio,
            maximum_ring_plane_distance=self.maximum_ring_plane_distance,
            include_torsions=self.include_torsions_in_validity3D,
            consider_hydrogens=self.consider_hydrogens,
        )

        self.conf_metrics: List[Metric] = [self.validity3D]

        self.conf_metrics.append(StrainEnergy())

        self.valid_conf_metrics: List[Metric] = [
            Uniqueness3D(self.tfd_threshold),
            Diversity3D(),
        ]
        self.training_valid_conf_metrics: List[TrainingMetric] = [
            Novelty3D(self.tfd_threshold)
        ]

        self.results = {}

    def get_results_for_cel(self, cel: GeneratedCEL, average=True) -> Dict[str, Any]:

        self.initialize()

        for metric in self.graph_metrics:
            metric_name = metric.name
            logging.info(f"Computing {metric_name}")
            self.results[metric_name] = metric.get(cel, average=average)

        if self.training_cel is not None:
            for metric in self.training_graph_metrics:
                metric_name = metric.name
                logging.info(f"Computing {metric_name}")
                self.results[metric_name] = metric.get(
                    cel=cel,
                    training_cel=self.training_cel,
                    average=average,
                )

        for metric in self.conf_metrics:
            metric_name = metric.name
            logging.info(f"Computing {metric_name}")
            self.results[metric_name] = metric.get(cel, average=average)

        self.results["Number of tested confs"] = cel.n_total_confs

        logging.info("Compute valid CEL for further analysis")
        valid_conf_ids = self.validity3D.valid_conf_ids
        valid_cel = GeneratedCEL.get_cel_subset(cel=cel, subset_conf_ids=valid_conf_ids)
        for metric in self.valid_conf_metrics:
            metric_name = metric.name
            logging.info(f"Computing {metric_name}")
            self.results[metric_name] = metric.get(cel=valid_cel, average=average)

        if self.training_cel is not None:
            for metric in self.training_valid_conf_metrics:
                metric_name = metric.name
                logging.info(f"Computing {metric_name}")
                self.results[metric_name] = metric.get(
                    cel=valid_cel,
                    training_cel=self.training_cel,
                    average=average,
                )

        # Looping here is not necessary, but could be used if you want to use different
        # reference geometry for different Validity3D
        validity3Ds = [
            metric for metric in self.conf_metrics if isinstance(metric, Validity3D)
        ]
        for validity3D in validity3Ds:

            self.results[f"Number of valid 3D confs ({validity3D.name})"] = (
                validity3D.n_valid_confs
            )

            n_inv_geom = validity3D.get_invalid_geometries()
            for geometry, values in n_inv_geom.items():
                self.results[f"Number of invalid {geometry}s ({validity3D.name})"] = (
                    values
                )

            new_patterns = validity3D.get_new_patterns()
            self.results[f"Number of new patterns ({validity3D.name})"] = len(
                new_patterns
            )
            self.results[f"Number of new unique patterns ({validity3D.name})"] = len(
                set(new_patterns)
            )

            geometry_combos = [
                ["bond"],
                ["angle"],
                ["torsion"],
                ["bond", "angle"],
                ["bond", "angle", "torsion"],
            ]

            for combo in geometry_combos:
                combo_str = "+".join(combo)
                min_q_values = validity3D.get_min_q_values_for_geometries(combo)
                self.results[f"Min {combo_str} q-value ({validity3D.name})"] = (
                    min_q_values
                )

                geo_mean_q_values = validity3D.get_geo_mean_q_values_for_geometries(
                    combo
                )
                self.results[
                    f"Geometric mean {combo_str} q-value ({validity3D.name})"
                ] = geo_mean_q_values

        # for k, v in self.results.items():
        #     if type(v) == np.int64:
        #         self.results[k] = int(v)
        #     elif type(v) == np.float64:
        #         self.results[k] = float(v)
        #     elif type(v) == np.ndarray:
        #         self.results[k] = v.tolist()
        #     elif type(v) == list:
        #         for i, item in enumerate(v):
        #             if type(item) == np.int64:
        #                 v[i] = int(item)
        #             elif type(item) == np.float64:
        #                 v[i] = float(item)

        return self.results

    def get_results_for_mol_list(
        self,
        mols: List[Mol],
        n_total_mols: int = None,
        names: list[str] = None,
        average: bool = True,
    ):
        cel = GeneratedCEL.from_mol_list(
            mol_list=mols, n_total_mols=n_total_mols, names=names
        )
        return self.get_results_for_cel(cel, average=average)

    def get_results_for_ase_db(
        self,
        filepath: str,
        cel_name: str = "generated_molecules",
        n_total_mols: int = None,
    ) -> Dict[str, Any]:
        reader = ASEDBReader()
        cel, self.n_total_mols = reader.read(filepath=filepath, cel_name=cel_name)
        if n_total_mols:  # User defined number of generated molecules
            self.n_total_mols = n_total_mols

        return self.get_results_for_cel(cel)

    def get_results_for_sdf_dir(
        self,
        dirpath: str,
        cel_name: str = "generated_molecules",
        n_total_mols: int = None,
    ) -> Dict[str, Any]:
        mols = []
        for filename in os.listdir(dirpath):
            filepath = os.path.join(dirpath, filename)
            if filepath.endswith(".sdf") and not filename.startswith("traj"):
                with Chem.SDMolSupplier(filepath) as suppl:
                    mols.extend([mol for mol in suppl])
        self.n_total_mols = len(mols)
        if n_total_mols is not None:  # User defined number of generated molecules
            self.n_total_mols = n_total_mols

        new_mols = []
        none_mol_counter = 0
        for mol in mols:
            if mol is None or mol.GetNumHeavyAtoms() == 0:
                none_mol_counter += 1
            else:
                new_mols.append(mol)
        mols = new_mols
        self.none_mol_counter = none_mol_counter

        cel = ConfEnsembleLibrary.from_mol_list(mol_list=mols, cel_name=cel_name)

        return self.get_results_for_cel(cel)
