import argparse
import itertools
import multiprocessing
import os
import pickle
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from rdkit import Chem

import flowr.scriptutil as util
from flowr.data.data_info import GeneralInfos as DataInfos
from flowr.eval.evaluate_interactions import compute_interaction_recovery_parallel
from flowr.eval.evaluate_util import (
    gather_predictions,
    gather_predictions_pilot,
)
from flowr.train import build_data_statistic
from flowr.util.metrics import (
    evaluate_gb3_validity,
    evaluate_gbsb3,
    evaluate_mol_metrics,
    evaluate_pb_validity,
    evaluate_posecheck,
    evaluate_statistics,
    evaluate_uniqueness,
    evaluate_validity,
)
from flowr.util.rdkit import sanitize_list

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _calc_sbdd_metrics(
    gen_ligs: list[Chem.Mol],
    ref_lig: Chem.Mol,
    ref_pdb: str,
):
    assert isinstance(
        gen_ligs, list
    ), "Input must be a list of molecules for one target."
    assert isinstance(ref_lig, Chem.Mol), "Reference ligand must be a molecule."
    assert isinstance(ref_pdb, str), "Reference PDB must be a string."

    gen_ligs = sanitize_list(gen_ligs, filter_uniqueness=False, sanitize=True)
    assert len(gen_ligs) > 0, "No valid/unique molecule found."

    # GB3-SBDD
    config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))
    gb3sb_metrics = evaluate_gbsb3(
        gen_ligs, ref_lig=ref_lig, pdb_file=ref_pdb, config=config, minimize=False
    )
    # PB-validity
    posebusters_validity = evaluate_pb_validity(
        gen_ligs, ref_lig=ref_lig, pdb_file=ref_pdb, config=config, minimize=False
    )
    # PoseCheck
    posecheck_metrics = evaluate_posecheck(gen_ligs, pdb_file=ref_pdb)

    sbdd_results = {
        **posebusters_validity,
        **gb3sb_metrics,
        **posecheck_metrics,
    }
    return sbdd_results


def _calc_sbdd_metrics_single(args):
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
    gen_ligs, ref_lig, ref_pdb = args
    # calc_sbdd_metrics returns a dictionary; here we compute metrics for 1 target.
    return _calc_sbdd_metrics(gen_ligs, ref_lig, str(ref_pdb))


def compute_sbdd_metrics_parallel(
    gen_ligs: list[list[Chem.Mol]],
    ref_ligs: list[Chem.Mol],
    ref_pdbs: list[str],
    num_workers=24,
):
    """
    Computes SBDD metrics for each target in parallel using one task per target.
    """
    if not (len(gen_ligs) == len(ref_ligs) == len(ref_pdbs)):
        raise ValueError("All lists must have the same length.")

    # Prepare one task per target.
    worker_args = list(zip(gen_ligs, ref_ligs, ref_pdbs))

    # Use Pool.map to distribute work. Pool.map automatically splits tasks among workers.
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(_calc_sbdd_metrics_single, worker_args)

    # Combine results across all targets.
    # Each result is a dictionary of metrics for one target.
    final_results = defaultdict(list)
    for result in results:
        if result is not None:
            for key, value in result.items():
                final_results[key].append(value)
    # Compute the mean over all targets.
    final_results = {key: np.nanmean(values) for key, values in final_results.items()}
    return final_results


# def calc_sbdd_metrics(
#     gen_ligs: list[list[Chem.Mol]],
#     ref_ligs: list[Chem.Mol],
#     ref_pdbs: list[str],
# ):
#     assert isinstance(
#         gen_ligs[0], list
#     ), "Input must be a list of lists of molecules - N ligands per target"

#     gen_ligs = [sanitize_list(mols, filter_uniqueness=False) for mols in gen_ligs]
#     assert (
#         min([len(ligs) for ligs in gen_ligs]) > 0
#     ), "No valid/unique molecule found for at least one of the targets."

#     # GB3-SBDD
#     config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))
#     gb3sb_metrics = evaluate_gbsb3_metrics(
#         gen_ligs, ref_ligs=ref_ligs, pdb_files=ref_pdbs, config=config, minimize=False
#     )
#     # PB-validity
#     posebusters_validity = evaluate_pb_validity(
#         gen_ligs, ref_ligs=ref_ligs, pdb_files=ref_pdbs, config=config, minimize=False
#     )
#     # PoseCheck
#     posecheck_metrics = evaluate_posecheck_metrics(gen_ligs, ref_pdbs=ref_pdbs)

#     sbdd_results = {**posebusters_validity, **gb3sb_metrics, **posecheck_metrics}
#     return sbdd_results


# def compute_sbdd_metrics_parallel(
#     gen_ligs: list[list[Chem.Mol]],
#     ref_ligs: list[Chem.Mol],
#     ref_pdbs: list[str],
#     n_procs=24,
# ):
#     """
#     Orchestrates the parallel computation of metrics.
#     `n_procs` is the number of processes for multiprocessing.
#     """

#     total_mols = len(gen_ligs)
#     if not (len(gen_ligs) == len(ref_ligs) == total_mols == len(ref_pdbs)):
#         raise ValueError("All lists must be the same length.")

#     gen_chunks = list(chunkify(gen_ligs, n_procs))
#     ref_chunks = list(chunkify(ref_ligs, n_procs))
#     pdb_chunks = list(chunkify(ref_pdbs, n_procs))

#     worker_args = [
#         (g_chunk, r_chunk, p_chunk)
#         for g_chunk, r_chunk, p_chunk in zip(gen_chunks, ref_chunks, pdb_chunks)
#     ]
#     with multiprocessing.Pool(processes=n_procs) as pool:
#         results = pool.starmap(calc_sbdd_metrics, worker_args)

#     assert len(results) == len(
#         gen_chunks
#     ), "Results length mismatch with number of processes"
#     final_results = defaultdict(list)
#     for result in results:
#         for key, value in result.items():
#             final_results[key].append(value)
#     final_results = {key: np.mean(value) for key, value in final_results.items()}
#     return final_results


def _eval_gb3_validity(args_tuple):
    """
    Helper function for parallelizing GB3 validity evaluation.
    args_tuple: (ligs, config)
    """
    ligs, config = args_tuple
    assert isinstance(ligs, list), "Input must be a list of molecules."
    return evaluate_gb3_validity(ligs, config=config)


def calc_gb3_metrics_parallel(gen_ligs: list[list[Chem.Mol]], num_workers: int):
    """
    Parallel version of calc_gb3_metrics.
    Expects gen_mols to be a list of list of molecules (one target per list).
    Uses num_workers processes to compute the GB3 validity metrics per target.
    Returns a dictionary of aggregated (mean and std) metrics.
    """
    # Load configuration once
    config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))

    # Create a list of arguments for each target
    arg_list = [(ligs, config) for ligs in gen_ligs]

    with multiprocessing.Pool(processes=num_workers) as pool:
        gb3_validity_list = pool.map(_eval_gb3_validity, arg_list)

    # Now aggregate the metrics across targets.
    final_results = defaultdict(list)
    for result in gb3_validity_list:
        for key, value in result.items():
            final_results[key].append(value)
    final_results_mean = {
        "GB3-" + key + " (mean)": np.nanmean(value)
        for key, value in final_results.items()
    }
    final_results_std = {
        "GB3-" + key + " (std)": np.nanstd(value)
        for key, value in final_results.items()
    }
    final_results = {**final_results_mean, **final_results_std}
    return final_results


def calc_gb3_metrics(gen_mols: list[list[Chem.Mol]]):
    assert isinstance(gen_mols[0], list), "Input must be a list of lists of molecules."

    config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))
    gb3_validity = [evaluate_gb3_validity(ligs, config=config) for ligs in gen_mols]

    valid3d = [v["Validity3D"] for v in gb3_validity]
    mean_valid3d = np.nanmean(valid3d)
    std_valid3d = np.nanstd(valid3d)

    unique2d = [v["Uniqueness2D"] for v in gb3_validity]
    mean_unique2d = np.nanmean(unique2d)
    std_unique2d = np.nanstd(unique2d)

    unique3d = [v["Uniqueness3D"] for v in gb3_validity]
    mean_unique3d = np.nanmean(unique3d)
    std_unique3d = np.nanstd(unique3d)

    diversity2d = [v["Diversity2D"] for v in gb3_validity]
    mean_diversity2d = np.nanmean(diversity2d)
    std_diversity2d = np.nanstd(diversity2d)

    diversity3d = [v["Diversity3D"] for v in gb3_validity]
    mean_diversity3d = np.nanmean(diversity3d)
    std_diversity3d = np.nanstd(diversity3d)

    strain_energy = [v["Strain energy"] for v in gb3_validity]
    mean_strain_energy = np.nanmean(strain_energy)
    std_strain_energy = np.nanstd(strain_energy)

    return {
        "Validity3D (mean)": mean_valid3d,
        "Validity3D (std)": std_valid3d,
        "Uniqueness2D (mean)": mean_unique2d,
        "Uniqueness2D (std)": std_unique2d,
        "Uniqueness3D (mean)": mean_unique3d,
        "Uniqueness3D (std)": std_unique3d,
        "Diversity2D (mean)": mean_diversity2d,
        "Diversity2D (std)": std_diversity2d,
        "Diversity3D (mean)": mean_diversity3d,
        "Diversity3D (std)": std_diversity3d,
        "StrainEnergy (mean)": mean_strain_energy,
        "StrainEnergy (std)": std_strain_energy,
    }


def calc_metrics(args, gen_mols):

    # data statistics
    print("Loading dataset statistics...")
    vocab = util.build_vocab()
    statistics = build_data_statistic(args)
    dataset_info = DataInfos(statistics, vocab, args)

    # load train smiles
    if Path(os.path.join(args.data_path, "train_mols.pkl")).exists():
        with open(os.path.join(args.data_path, "train_mols.pkl"), "rb") as f:
            train_mols = pickle.load(f)
    else:
        raise FileNotFoundError("Training mols not found.")
    train_smiles = [Chem.MolToSmiles(mol) for mol in train_mols]

    # evaluate validity on all sampled molecules
    validity = evaluate_validity(gen_mols)

    # evaluate mol properties and statistics on the subset of valid and unique molecules
    gen_mols = sanitize_list(gen_mols, sanitize=True, filter_uniqueness=False)
    mol_metrics = evaluate_mol_metrics(gen_mols, train_smiles=train_smiles)
    statistics = evaluate_statistics(gen_mols, state="test", dataset_info=dataset_info)

    mol_results = {**validity, **mol_metrics, **statistics}
    return mol_results


def main(args):
    #################################### Evaluate metrics ####################################
    print("Running evaluation...")

    if args.evaluate_pilot:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )

    else:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    if args.dataset == "crossdocked":
        ref_pdbs = [
            os.path.join(
                args.data_path, "test", Path(path).stem.split("_with_hs")[0] + ".pdb"
            )
            for path in ref_pdbs
        ]

    start_time = time.time()
    print(f"Computing molecule metrics on {args.num_workers} CPUs...")
    print(
        f"Calculating metrics for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Uniqueness
    uniqueness = [evaluate_uniqueness(ligs) for ligs in gen_ligs]
    uniqueness = {
        "Uniqueness (mean)": np.mean(uniqueness),
        "Uniqueness (std)": np.std(uniqueness),
    }
    # GenBench3D
    gb3_metrics = calc_gb3_metrics_parallel(gen_ligs, num_workers=args.num_workers)
    print(
        f"GB3 metrics computed in {round((time.time() - start_time) / 60, 2)} minutes."
    )

    # Other molecular metrics
    gen_mols = list(itertools.chain(*gen_ligs))
    results_mol = calc_metrics(args, gen_mols)
    results_mol = {**results_mol, **uniqueness, **gb3_metrics}
    print(
        f"Molecule metrics computed in {round((time.time() - start_time) / 60, 2)} minutes."
    )

    # SBDD metrics
    start_time = time.time()
    print(f"Computing SBDD metrics on {args.num_workers} CPUs...")
    results_sbdd = compute_sbdd_metrics_parallel(
        gen_ligs, ref_ligs, ref_pdbs, num_workers=args.num_workers
    )
    print(
        f"SBDD metrics computed in {round((time.time() - start_time) / 60, 2)} minutes."
    )
    results = {**results_mol, **results_sbdd}

    if not args.dataset == "crossdocked":
        ### CHECK INTERACTION RECOVERY ###
        start_time = time.time()
        print(f"Computing interaction recovery on {args.num_workers} CPUs...")
        interaction_recovery = compute_interaction_recovery_parallel(
            args, gen_ligs, ref_ligs, ref_pdbs, n_procs=args.num_workers
        )
        print(
            f"Interaction recovery {interaction_recovery} computed in {round((time.time() - start_time) / 60, 2)} minutes."
        )
        results = {**results, **interaction_recovery}

    # Save results
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.valid_unique:
        torch.save(results, Path(args.save_dir) / "metrics_valid_unique.pt")
    else:
        torch.save(results, Path(args.save_dir) / "metrics.pt")
    util.print_results(results)

    print("Evaluation complete. Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--return_interaction_list", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")

    args = parser.parse_args()

    main(args)
