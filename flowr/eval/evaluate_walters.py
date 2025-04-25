import argparse
import itertools
import multiprocessing
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import useful_rdkit_utils as uru
from rdkit import Chem
from rdkit.Chem import PandasTools

from flowr.eval.evaluate_dataset import load_data
from flowr.eval.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.scriptutil import print_results
from flowr.util.rdkit import sanitize_list, write_sdf_file


def evaluate_walters(mols, return_smarts=False, verbose=False):
    # Create a directory to save the filtered SDF files
    results = defaultdict()
    with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
        sdf_path = tmp.name
        write_sdf_file(sdf_path, mols)
        df_0 = PandasTools.LoadSDF(sdf_path)

    # number of fragments
    df_0["num_frags"] = df_0.ROMol.apply(uru.count_fragments)
    df_1 = df_0.query("num_frags == 1").copy()
    num_frags = len(df_0) - len(df_1)
    results["num_frags"] = num_frags
    if verbose:
        print(f"Number of fragments: {num_frags}")

    # InChi duplicates
    df_1["inchi"] = df_1.ROMol.apply(Chem.MolToInchiKey)
    df_2 = df_1.drop_duplicates(subset="inchi").copy()
    num_duplicates = len(df_1) - len(df_2)
    results["num_duplicates"] = num_duplicates
    results["num_unique"] = len(df_2)
    if verbose:
        print(f"Duplicates removed: {num_duplicates}")

    # remove odd ring systems
    ring_system_lookup = uru.RingSystemLookup()
    df_2["ring_systems"] = [list(ring_system_lookup.process_mol(x)) for x in df_2.ROMol]
    df_2[["min_ring", "min_freq"]] = [
        list(uru.get_min_ring_frequency(x)) for x in df_2.ring_systems
    ]
    df_3 = df_2.query("min_freq >= 100").copy()
    num_odd_ring_systems = len(df_2) - len(df_3)
    results["num_odd_ring_systems"] = num_odd_ring_systems
    results["perc_odd_ring_systems"] = round(
        100 * (num_odd_ring_systems / len(df_2)), 2
    )
    if verbose:
        print(f"Odd ring systems removed: {num_odd_ring_systems}")

    # remove reactive or odd functional groups with REOS rules
    smarts = defaultdict(list)
    list_of_rules = ["Dundee", "Glaxo", "LINT", "PAINS", "BMS", "SureChEMBL"]
    for rule in list_of_rules:
        reos = uru.REOS()
        reos.set_output_smarts(True)
        reos.set_active_rule_sets([rule])
        df_reos = df_2.copy()
        df_reos[["rule_set", "reos", "smarts"]] = [
            list(reos.process_mol(x)) for x in df_2.ROMol
        ]
        smarts[rule].extend(df_reos.query("reos != 'ok'").copy()["smarts"].tolist())
        df_fgroups = df_reos.query("reos == 'ok'").copy()
        num_odd_fgroups = len(df_reos) - len(df_fgroups)
        results[f"num_odd_fgroups_{rule}"] = num_odd_fgroups
        results[f"perc_odd_fgroups_{rule}"] = round(
            100 * (num_odd_fgroups / len(df_2)), 2
        )
        if verbose:
            print(
                f"Number of odd/reactive functional groups with rule {rule}: {num_odd_fgroups}"
            )
    if return_smarts:
        results["smarts"] = smarts
    return results


def create_list_defaultdict():
    return defaultdict(list)


def _calc_walters_metrics(
    gen_ligs: list[Chem.Mol],
    ref_pdb: str,
):
    assert isinstance(
        gen_ligs, list
    ), "Input must be a list of molecules for one target."
    if len(gen_ligs) == 0:
        return {}

    assert isinstance(ref_pdb, str), "Reference PDB must be a string."

    # walters
    results = evaluate_walters(gen_ligs, verbose=False)
    return results


def _calc_walters_metrics_single(args):
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
    gen_ligs, ref_pdb = args
    # calc_sbdd_metrics returns a dictionary; here we compute metrics for 1 target.
    results = _calc_walters_metrics(gen_ligs, str(ref_pdb))
    return {f"{Path(ref_pdb).stem}": results}


def calc_walters_metrics_parallel(
    gen_ligs: list[list[Chem.Mol]],
    ref_pdbs: list[str],
    num_workers=24,
    return_list=False,
):
    """
    Computes SBDD metrics for each target in parallel using one task per target.
    """
    if not (len(gen_ligs) == len(ref_pdbs)):
        raise ValueError("All lists must have the same length.")

    # Prepare one task per target.
    worker_args = list(zip(gen_ligs, ref_pdbs))

    # Use Pool.map to distribute work. Pool.map automatically splits tasks among workers.
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_calc_walters_metrics_single, worker_args)

    # Combine results across all targets.
    # Each result is a dictionary of metrics for one target.
    final_results = defaultdict(create_list_defaultdict)
    for result in results:
        if result is not None:
            for target, values in result.items():
                final_results[target] = values
    if return_list:
        return dict(final_results)
    # Compute the final dict that only contains walters scores mean and mean std aggregated over all targets
    final_results_dict = defaultdict(list)
    for target, values in final_results.items():
        for key, value in values.items():
            final_results_dict[key].append(value)
    final_results = {
        key + " (mean)": np.nanmean(values)
        for key, values in final_results_dict.items()
    }
    final_results.update(
        {
            key + " (std)": np.nanstd(values)
            for key, values in final_results_dict.items()
        }
    )
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--return_list", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    parser.add_argument("--evaluate_test_data", action="store_true")
    parser.add_argument("--evaluate_train_data", action="store_true")
    parser.add_argument("--evaluate_val_data", action="store_true")
    args = parser.parse_args()

    # Load the data
    if args.evaluate_pilot:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )
    elif args.evaluate_train_data:
        ligs, ref_ligs, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="train"
        )
        gen_ligs = [[lig] for lig in ligs]
    elif args.evaluate_val_data:
        ligs, ref_ligs, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="val"
        )
        gen_ligs = [[lig] for lig in ligs]
    elif args.evaluate_test_data:
        ligs, ref_ligs, ref_pdbs = load_data(
            data_path=args.data_path, remove_hs=True, state="test"
        )
        gen_ligs = [[lig] for lig in ligs]
    else:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    start_time = time.time()
    print(f"Computing Walters metrics on {args.num_workers} CPUs...")
    print(
        f"Computing Walters metrics for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Compute the metrics
    gen_ligs_san = [
        sanitize_list(mols, filter_uniqueness=False, sanitize=True) for mols in gen_ligs
    ]
    walters_metrics = calc_walters_metrics_parallel(
        gen_ligs_san,
        ref_pdbs,
        num_workers=args.num_workers,
        return_list=args.return_list,
    )
    # Save the results
    if args.evaluate_test_data or args.evaluate_val_data or args.evaluate_train_data:
        state = (
            "test"
            if args.evaluate_test_data
            else "val" if args.evaluate_val_data else "train"
        )
        save_path = os.path.join(args.save_dir, f"walters_metrics_dict_{state}.pt")
    else:
        save_path = os.path.join(args.save_dir, "walters_metrics_dict.pt")
    torch.save(walters_metrics, save_path)
    if not args.return_list:

        print("Walters metrics:\n")
        print_results(walters_metrics)
    print(f"Walters metrics saved to {save_path}")
