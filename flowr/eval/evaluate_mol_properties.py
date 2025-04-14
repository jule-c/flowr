import argparse
import itertools
import multiprocessing
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from flowr.evaluate_dataset import load_data
from flowr.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.util.metrics import (
    calculate_hacceptors,
    calculate_hdonors,
    calculate_lipinski,
    calculate_logp,
    calculate_qed,
    calculate_rotatable_bonds,
    calculate_sa,
    calculate_tpsa,
    num_aromatic_rings,
    num_rings,
)
from flowr.util.rdkit import sanitize_list


def create_list_defaultdict():
    return defaultdict(list)


def evaluate_mol_properties(gen_mols, return_list=False):
    """
    Evaluate the molecular properties of the generated molecules.
    """
    mol_properties_metrics = defaultdict(list)
    mol_properties_metrics["logP"] = [calculate_logp(mol) for mol in gen_mols]
    mol_properties_metrics["HBA"] = [calculate_hacceptors(mol) for mol in gen_mols]
    mol_properties_metrics["HBD"] = [calculate_hdonors(mol) for mol in gen_mols]
    mol_properties_metrics["TPSA"] = [calculate_tpsa(mol) for mol in gen_mols]
    mol_properties_metrics["SA"] = [calculate_sa(mol) for mol in gen_mols]
    mol_properties_metrics["Lipinski"] = [calculate_lipinski(mol) for mol in gen_mols]
    mol_properties_metrics["QED"] = [calculate_qed(mol) for mol in gen_mols]
    mol_properties_metrics["rings"] = [num_rings(mol) for mol in gen_mols]
    mol_properties_metrics["aromatic_rings"] = [
        num_aromatic_rings(mol) for mol in gen_mols
    ]
    mol_properties_metrics["rot_bonds"] = [
        calculate_rotatable_bonds(mol) for mol in gen_mols
    ]
    if return_list:
        return mol_properties_metrics
    mol_properties_mean = {
        key + "( mean)": np.nanmean(values)
        for key, values in mol_properties_metrics.items()
    }
    mol_properties_std = {
        key + "( std)": np.nanstd(values)
        for key, values in mol_properties_metrics.items()
    }
    mol_properties_metrics = {**mol_properties_mean, **mol_properties_std}
    return mol_properties_metrics


def _calc_mol_properties_metrics_single(args):
    """
    Calculate the mol properties metrics for all sampled molecules.
    """
    gen_mols, ref_pdb, return_list = args
    mol_properties_metrics = evaluate_mol_properties(gen_mols, return_list=return_list)
    return {f"{Path(ref_pdb).stem}": mol_properties_metrics}


def calc_mol_properties_metrics_parallel(
    gen_ligs: list[list[Chem.Mol]],
    ref_pdbs: list[str],
    return_list=False,
    num_workers=24,
):
    """
    Computes molecular metrics
    """
    worker_args = list(zip(gen_ligs, ref_pdbs, [return_list] * len(gen_ligs)))
    # Use Pool.map to distribute work. Pool.map automatically splits tasks among workers.
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_calc_mol_properties_metrics_single, worker_args)

    # Combine results across all targets.
    # Each result is a dictionary of metrics for one target.
    final_results = defaultdict(create_list_defaultdict)
    for result in results:
        if result is not None:
            for target, values in result.items():
                for key, value in values.items():
                    if isinstance(value, list):
                        final_results[target][key].extend(value)
                    else:
                        final_results[target][key].append(value)
    if return_list:
        return dict(final_results)
    # Compute the final dict that only contains Vina scores mean and mean std aggregated over all targets
    final_results_dict = {"Vina scores (mean)": {}, "Vina scores (std)": {}}
    for target, values in final_results.items():
        for key, value in values.items():
            final_results_dict[key].append(value)
    final_results = {
        key: np.nanmean(values) for key, values in final_results_dict.items()
    }
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--return_list", action="store_true")
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    parser.add_argument("--evaluate_test_data", action="store_true")
    parser.add_argument("--evaluate_train_data", action="store_true")
    parser.add_argument("--evaluate_val_data", action="store_true")
    args = parser.parse_args()

    # Load the data
    if args.evaluate_pilot:
        gen_ligs, _, _, _, ref_pdbs = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )
    elif args.evaluate_train_data:
        ligs, _, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="train"
        )
        gen_ligs = [[lig] for lig in ligs]
    elif args.evaluate_val_data:
        ligs, _, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="val"
        )
        gen_ligs = [[lig] for lig in ligs]
    elif args.evaluate_test_data:
        ligs, _, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="test"
        )
        gen_ligs = [[lig] for lig in ligs]
    else:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    start_time = time.time()
    print(f"Computing molecule metrics on {args.num_workers} CPUs...")
    print(
        f"Calculating metrics for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Compute the metrics
    gen_ligs_san = [
        sanitize_list(mols, filter_uniqueness=True, sanitize=True) for mols in gen_ligs
    ]
    mol_properties_metrics = calc_mol_properties_metrics_parallel(
        gen_ligs_san,
        ref_pdbs,
        return_list=args.return_list,
        num_workers=args.num_workers,
    )
    # Save the results
    if args.evaluate_test_data or args.evaluate_val_data or args.evaluate_train_data:
        state = (
            "test"
            if args.evaluate_test_data
            else "val" if args.evaluate_val_data else "train"
        )
        save_path = os.path.join(
            args.save_dir, f"mol_properties_metrics_dict_{state}.pt"
        )
    else:
        save_path = os.path.join(args.save_dir, "mol_properties_metrics_dict.pt")
    torch.save(mol_properties_metrics, save_path)
    if not args.return_list:
        print(f"Molecular properties: {mol_properties_metrics}")
    print(f"Molecular properties saved to {save_path}")
