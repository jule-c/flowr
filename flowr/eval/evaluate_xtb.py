import argparse
import itertools
import os
import re
import signal
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from tqdm import tqdm

from flowr.evaluate_dataset import load_data
from flowr.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.util.rdkit import sanitize_list

signal.signal(signal.SIGTERM, signal.SIG_DFL)


def create_list_defaultdict():
    return defaultdict(list)


def split_list(data, num_chunks):
    chunk_size = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        chunk_end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:chunk_end])
        start = chunk_end
    return chunks


def evaluate_energy(lig, return_list=False, relax=False):
    """
    Calculate energy of generated ligands.
    If relax is True, compute the relaxation energy difference,
    i.e. (predicted energy - relaxed energy).

    Parameters:
      lig: a Chem.Mol molecule
      return_list: if True, return a list of energies per molecule,
                   otherwise return a dict with mean and std.
      relax: if True, evaluate the relaxed energy instead of just total energy.
    """
    mol_copy = Chem.Mol(lig)
    mol_copy = Chem.AddHs(mol_copy, addCoords=True)
    try:
        # Convert molecule to XYZ format
        xyz_block = Chem.MolToXYZBlock(mol_copy)
        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "temp.xyz")
            with open(xyz_path, "w") as f:
                f.write(xyz_block)
            # Run xTB to compute the predicted (non-relaxed) energy
            pred_command = ["xtb", xyz_path, "--gfn", "2", "--alpb", "water", "--sp"]
            proc_pred = subprocess.run(pred_command, capture_output=True, text=True)
            if proc_pred.returncode != 0:
                print(f"xTB (predicted) command failed with error:\n{proc_pred.stderr}")
                return np.nan
            predicted_energy = np.nan
            for line in proc_pred.stdout.splitlines():
                if "total energy" in line.lower():
                    try:
                        matches = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                        if matches:
                            predicted_energy = float(matches[0])
                            if relax:
                                predicted_energy *= (
                                    627.509  # Convert from Hartree to kcal/mol
                                )
                        else:
                            print(f"No energy value found in line: {line}")
                            predicted_energy = np.nan
                    except Exception:
                        predicted_energy = np.nan
                    break

            if relax:
                # Run xTB for relaxed energy using optimization (--opt)
                relax_command = [
                    "xtb",
                    xyz_path,
                    "--gfn",
                    "2",
                    "--alpb",
                    "water",
                    "--opt",
                ]
                proc_relax = subprocess.run(
                    relax_command, capture_output=True, text=True
                )
                if proc_relax.returncode != 0:
                    print(
                        f"xTB (relaxed) command failed with error:\n{proc_relax.stderr}"
                    )
                    return np.nan
                # Instead of taking the first occurrence, collect all TOTAL ENERGY matches and use the last one.
                energy_matches = re.findall(
                    r"TOTAL ENERGY\s+([-+]?\d*\.\d+)\s",
                    proc_relax.stdout,
                )
                if energy_matches:
                    final_relaxed_energy = float(energy_matches[-1]) * 627.509
                else:
                    print("No relaxed energy found in xTB output after convergence.")
                    return np.nan
                return abs(predicted_energy - final_relaxed_energy)
            else:
                return predicted_energy

    except Exception as e:
        print(f"Error calculating energy for molecule: {e}")
        return np.nan


def _calc_energy_metrics(gen_ligs, ref_pdb, return_list, relax=False):
    """
    Helper function to compute SBDD metrics for a single target.
    """
    energies = []
    for mol in tqdm(
        gen_ligs,
        desc=f"Evaluating ligands for target {Path(ref_pdb).stem}",
        leave=False,
        total=len(gen_ligs),
        position=1,
    ):
        energy = evaluate_energy(mol, return_list=return_list, relax=relax)
        energies.append(energy)

    if return_list:
        energy_results = energies
    else:
        energy_results = {
            "Energy (mean)": np.nanmean(energies),
            "Energy (std)": np.nanstd(energies),
        }
    return {f"{Path(ref_pdb).stem}": energy_results}


def calc_energy_metrics(
    gen_ligs: list[list[Chem.Mol]],
    ref_pdbs: list[str],
    return_list: bool = False,
    relax: bool = False,
):
    """
    Computes Energy for each target in parallel.
    """
    if not (len(gen_ligs) == len(ref_pdbs)):
        raise ValueError("All lists must have the same length.")

    final_results = (
        defaultdict(list) if return_list else defaultdict(create_list_defaultdict)
    )
    for gen_ligs_target, ref_pdb in tqdm(
        zip(gen_ligs, ref_pdbs), total=len(gen_ligs), desc="Processing targets"
    ):
        result = _calc_energy_metrics(
            gen_ligs_target, ref_pdb, return_list=return_list, relax=relax
        )
        if result is not None:
            target = f"{Path(ref_pdb).stem}"
            values = result[target]
            if isinstance(values, list):
                final_results[target].extend(values)
            elif isinstance(values, dict):
                for key, value in values.items():
                    final_results[target][key].append(value)
    if return_list:
        return dict(final_results)
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
    parser.add_argument("--mp_index", type=int, default=1)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--return_list", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--calc_relax_energy", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    parser.add_argument("--evaluate_test_data", action="store_true")
    parser.add_argument("--evaluate_train_data", action="store_true")
    parser.add_argument("--evaluate_val_data", action="store_true")
    args = parser.parse_args()

    if args.num_jobs == 1:
        print("Running in single process mode.")
        assert args.mp_index == 1, "mp_index must be 1 in single process mode."

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

    # Split the data
    gen_ligs = split_list(gen_ligs, args.num_jobs)[args.mp_index - 1]
    ref_pdbs = split_list(ref_pdbs, args.num_jobs)[args.mp_index - 1]
    assert len(gen_ligs) == len(
        ref_pdbs
    ), "Number of generated ligands and reference PDBs must be the same."

    start_time = time.time()
    print(f"Computing Energy metrics on worker {args.mp_index}.")
    print(
        f"Computing Energy metrics for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Compute the metrics
    gen_ligs_san = [
        sanitize_list(mols, filter_uniqueness=False, sanitize=True) for mols in gen_ligs
    ]
    energy_metrics = calc_energy_metrics(
        gen_ligs_san,
        ref_pdbs,
        return_list=args.return_list,
        relax=args.calc_relax_energy,
    )

    print(
        f"Energy metrics computed in {time.time() - start_time:.2f} seconds on worker {args.mp_index}."
    )
    # Save the results
    job_id = f"_{args.mp_index}" if args.num_jobs > 1 else ""
    if args.evaluate_test_data or args.evaluate_val_data or args.evaluate_train_data:
        state = (
            "test"
            if args.evaluate_test_data
            else "val" if args.evaluate_val_data else "train"
        )
        if args.calc_relax_energy:
            save_path = os.path.join(
                args.save_dir, f"relax_energy_metrics_dict_{state}{job_id}.pt"
            )
        else:
            save_path = os.path.join(
                args.save_dir, f"energy_metrics_dict_{state}{job_id}.pt"
            )
    else:
        if args.calc_relax_energy:
            save_path = os.path.join(
                args.save_dir, f"relax_energy_metrics_dict{job_id}.pt"
            )
        else:
            save_path = os.path.join(args.save_dir, f"energy_metrics_dict{job_id}.pt")
    torch.save(energy_metrics, save_path)
    if not args.return_list:
        print(f"Energy metrics: {energy_metrics}")
    print(f"Energy metrics saved to {save_path}")

    # print mean results
    total_ligs = 0
    mean_results = []
    std_results = []
    if args.return_list:
        for _, values in energy_metrics.items():
            total_ligs += len(values)
            mean_results.append(np.nanmean(values))
            std_results.append(np.nanstd(values))
        print(
            f"Calculated mean and std total energy for {total_ligs} ligands across {len(mean_results)} targets."
        )
        print(f"Mean Energy (Eh): {np.mean(mean_results)} +- {np.mean(std_results)}")
    else:
        print(f"Results per target: {energy_metrics}")
