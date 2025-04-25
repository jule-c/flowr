import argparse
import itertools
import multiprocessing
import signal
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from flowr.eval.evaluate_util import (
    chunkify,
    gather_predictions,
    gather_predictions_pilot,
)
from flowr.util.metrics import (
    evaluate_interaction_recovery,
)
from flowr.util.rdkit import sanitize_list

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def calc_interaction_recovery(gen_mols, ref_mols, ref_pdbs, args):
    """
    Worker function to compute a partial InteractionRecovery metric
    on a chunk of the data.
    """

    assert len(ref_mols) == len(ref_pdbs)

    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    gen_mols = [sanitize_list(mols, filter_uniqueness=False) for mols in gen_mols]
    if len(gen_mols) == 0:
        print(
            "No valid/unique molecules found for all of the targets in the current batch. Probably something is off!"
        )
        return
    failed_ids = [i for i, mols in enumerate(gen_mols) if mols is None]
    gen_mols = [mols for mols in gen_mols if mols is not None]
    ref_mols = [mols for i, mols in enumerate(ref_mols) if i not in failed_ids]
    ref_pdbs = [p for i, p in enumerate(ref_pdbs) if i not in failed_ids]

    out = evaluate_interaction_recovery(
        gen_mols,
        ref_mols,
        ref_pdbs,
        add_optimize_gen_lig_hs=args.remove_hs,
        add_optimize_ref_lig_hs=args.remove_hs,
        optimize_pocket_hs=False,
        process_pocket=False,
        optimization_method="prolif_mmff",
        pocket_cutoff=6.0,
        strip_invalid=True,
        save_dir=args.save_dir,
        return_list=args.return_interaction_list,
    )
    return out


def compute_interaction_recovery_parallel(
    args, gen_ligs, ref_ligs_with_hs, ref_pdbs_with_hs, n_procs=24
):
    """
    Orchestrates the parallel computation of interaction recovery.
    `n_procs` is the number of processes for multiprocessing.
    """

    total_batches = len(gen_ligs)
    total_mols = sum([len(ligs) for ligs in gen_ligs])
    if not (len(ref_ligs_with_hs) == total_batches == len(ref_pdbs_with_hs)):
        raise ValueError("All lists must have the same length.")

    gen_chunks = list(chunkify(gen_ligs, n_procs))
    ref_chunks = list(chunkify(ref_ligs_with_hs, n_procs))
    pdb_chunks = list(chunkify(ref_pdbs_with_hs, n_procs))

    assert len(gen_chunks) == len(ref_chunks) == len(pdb_chunks)

    worker_args = [
        (g_chunk, r_chunk, p_chunk, args)
        for g_chunk, r_chunk, p_chunk in zip(gen_chunks, ref_chunks, pdb_chunks)
    ]
    with multiprocessing.Pool(processes=n_procs) as pool:
        results = pool.starmap(calc_interaction_recovery, worker_args)

    assert len(results) == len(
        gen_chunks
    ), "Partial results length mismatch with number of processes"

    if args.return_interaction_list:
        recovery_rates = defaultdict(list)
        tanimoto_sims = defaultdict(list)
        for result in results:
            if result is not None:
                rr_results = result["Recovery rate"]
                for k, v in rr_results.items():
                    v = [x for x in v if x is not None]
                    recovery_rates[k].extend(v)
                ts_results = result["Tanimoto similarity"]
                for k, v in ts_results.items():
                    v = [x for x in v if x is not None]
                    tanimoto_sims[k].extend(v)
        return {
            "PLIF recovery rate": recovery_rates,
            "PLIF Tanimoto similarity": tanimoto_sims,
        }

    random_fail = sum([st is None for st in results])
    compute_fail = sum(
        [st["Number of failed molecules"] for st in results if st is not None]
    )
    total = sum([st["Number of tested molecules"] for st in results if st is not None])
    recovery_rate_mean = [
        st["PLIF recovery (mean)"] for st in results if st is not None
    ]
    recovery_rate_std = [st["PLIF recovery (std)"] for st in results if st is not None]
    tanimoto_sim_mean = [
        st["PLIF Tanimoto similarity (mean)"] for st in results if st is not None
    ]
    tanimoto_sim_std = [
        st["PLIF Tanimoto similarity (std)"] for st in results if st is not None
    ]

    mean_recovery_rate = np.nanmean(recovery_rate_mean)
    std_recovery_rate = np.nanmean(recovery_rate_std)
    mean_tanimoto_sim = np.nanmean(tanimoto_sim_mean)
    std_tanimoto_sim = np.nanmean(tanimoto_sim_std)

    return {
        "PLIF recovery rate (mean)": mean_recovery_rate,
        "PLIF recovery rate (std)": std_recovery_rate,
        "PLIF Tanimoto similarity (mean)": mean_tanimoto_sim,
        "PLIF Tanimoto similarity (std)": std_tanimoto_sim,
        "Number of molecules": int(total_mols),
        "Number of tested molecules": int(total),
        "Number of failed molecules": int(random_fail + compute_fail),
    }


def main(args):
    if args.evaluate_pilot:
        gen_ligs, _, ref_ligs_with_hs, _, ref_pdbs_with_hs = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )
    else:
        gen_ligs, _, ref_ligs_with_hs, _, ref_pdbs_with_hs = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    print(
        f"Calculating interactions for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    ### CHECK INTERACTION RECOVERY ###
    start_time = time.time()
    print(f"Computing interaction recovery on {args.num_workers} CPUs...")
    interaction_recovery = compute_interaction_recovery_parallel(
        args, gen_ligs, ref_ligs_with_hs, ref_pdbs_with_hs, n_procs=args.num_workers
    )
    if not args.return_interaction_list:
        print(
            f"Interaction recovery: {interaction_recovery['PLIF recovery rate (mean)']} +- {interaction_recovery['PLIF recovery rate (std)']} computed in {round((time.time() - start_time) / 60, 2)} minutes."
        )
    else:
        rr_mean = np.nanmean(
            [np.nanmean(v) for v in interaction_recovery["PLIF recovery rate"].values()]
        )
        rr_std = np.nanmean(
            [np.nanstd(v) for v in interaction_recovery["PLIF recovery rate"].values()]
        )
        ts_mean = np.nanmean(
            [
                np.nanmean(v)
                for v in interaction_recovery["PLIF Tanimoto similarity"].values()
            ]
        )
        ts_std = np.nanmean(
            [
                np.nanstd(v)
                for v in interaction_recovery["PLIF Tanimoto similarity"].values()
            ]
        )
        print(
            f"Interaction recovery rate: {rr_mean} +- {rr_std} computed in {round((time.time() - start_time) / 60, 2)} minutes."
        )
        print(
            f"Interaction Tanimoto similarity: {ts_mean} +- {ts_std} computed in {round((time.time() - start_time) / 60, 2)} minutes."
        )
    # Save recovery_rate
    save_dir = Path(args.save_dir)
    lst = "" if not args.return_interaction_list else "_list"
    if args.valid_unique:
        torch.save(
            interaction_recovery,
            save_dir / f"interaction_recovery_valid_unique{lst}.pt",
        )
    else:
        torch.save(interaction_recovery, save_dir / f"interaction_recovery{lst}.pt")

    print("Evaluation complete. Script finished.")


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--return_interaction_list", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    args = parser.parse_args()

    main(args)
