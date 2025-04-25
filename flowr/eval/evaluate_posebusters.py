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

from flowr.eval.evaluate_dataset import load_data
from flowr.eval.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.util.metrics import evaluate_pb_validity
from flowr.util.rdkit import sanitize_list


def create_list_defaultdict():
    return defaultdict(list)


def _calc_posebusters_metrics(
    gen_ligs: list[Chem.Mol],
    ref_pdb: str,
    return_list=False,
):
    assert isinstance(
        gen_ligs, list
    ), "Input must be a list of molecules for one target."
    if len(gen_ligs) == 0:
        return [] if return_list else {}

    assert isinstance(ref_pdb, str), "Reference PDB must be a string."

    # PoseBusters
    posebusters_metrics = evaluate_pb_validity(
        gen_ligs, pdb_file=ref_pdb, return_list=return_list
    )
    return posebusters_metrics


def _calc_posebusters_metrics_single(args):
    """
    Helper function to compute SBDD metrics for a single target.
    Each argument is expected to be a tuple:
      (gen_ligs_target, ref_lig, ref_pdb)
    where:
      gen_ligs_target: a list of generated molecules for one target
      ref_lig: the reference ligand (Chem.Mol) for that target
      ref_pdb: the reference pdb filename as a string
    Since calc_sbdd_metrics expects lists of lists, we wrap gen_ligs_target,
    ref_lig and ref_pdb in a list.
    """
    gen_ligs, ref_pdb, return_list = args
    # calc_sbdd_metrics returns a dictionary; here we compute metrics for 1 target.
    pb_results = _calc_posebusters_metrics(
        gen_ligs, str(ref_pdb), return_list=return_list
    )
    return {f"{Path(ref_pdb).stem}": pb_results}


def calc_posebusters_metrics_parallel(
    gen_ligs: list[list[Chem.Mol]],
    ref_pdbs: list[str],
    return_list: bool = False,
    num_workers=24,
):
    """
    Computes SBDD metrics for each target in parallel using one task per target.
    """
    if not (len(gen_ligs) == len(ref_pdbs)):
        raise ValueError("All lists must have the same length.")

    # Prepare one task per target.
    worker_args = list(zip(gen_ligs, ref_pdbs, [return_list] * len(gen_ligs)))

    # Use Pool.map to distribute work. Pool.map automatically splits tasks among workers.
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_calc_posebusters_metrics_single, worker_args)

    # Combine results across all targets.
    # Each result is a dictionary of metrics for one target.
    if return_list:
        final_results = defaultdict(list)
        for result in results:
            if result is not None:
                for target, values in result.items():
                    final_results[target].extend(values)
        return final_results

    final_results = defaultdict(create_list_defaultdict)
    for result in results:
        if result is not None:
            for target, values in result.items():
                for key, value in values.items():
                    if isinstance(value, list):
                        final_results[target][key].extend(value)
                    else:
                        final_results[target][key].append(value)
    # Compute the final dict that only contains Vina scores mean and mean std aggregated over all targets
    final_results_dict = defaultdict(list)
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
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--return_list", action="store_true")
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
        gen_ligs, _, _, _, ref_pdbs = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    start_time = time.time()
    print(f"Computing PoseBusters metrics on {args.num_workers} CPUs...")
    print(
        f"Computing PoseBusters metrics for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Compute the metrics
    gen_ligs_san = [
        sanitize_list(mols, filter_uniqueness=False, sanitize=True) for mols in gen_ligs
    ]
    posebusters_metrics = calc_posebusters_metrics_parallel(
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
        save_path = os.path.join(args.save_dir, f"posebusters_metrics_dict_{state}.pt")
    else:
        save_path = os.path.join(args.save_dir, "posebusters_metrics_dict.pt")
    torch.save(posebusters_metrics, save_path)
    if not args.return_list:
        print(f"PoseBusters metrics: {posebusters_metrics}")
    print(f"PoseBusters metrics saved to {save_path}")
