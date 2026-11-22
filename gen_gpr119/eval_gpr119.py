"""Evaluate one generate_from_pdb run (single target) with the standard FLOWR metrics.

generate_from_pdb saves samples_<target>.pt, while flowr.eval.evaluate_metrics
expects a predictions.pt in the multiple-file batch layout, referencing
reference PDBs under <data_path>/ref_test_pdbs. This wrapper bridges the two:
it rebuilds predictions.pt from the samples file, copies the reference PDBs
into <data_path>/ref_test_pdbs, runs the standard evaluation, and stores the
result as metrics_gpr119.pt next to the samples.

Usage (from ~/flowr):
    python gen_gpr119/eval_gpr119.py \
        --save_dir data/gpr119/exp/out_G3 \
        --pred_file data/gpr119/exp/out_G3/samples_unfiltered_GPR119_protein.pt \
        --data_path data/spindr --num_workers 12 --remove_hs \
        --return_interaction_list
"""

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch

from flowr.eval import evaluate_metrics


def find_samples(pred_file, save_dir):
    if pred_file is not None:
        return Path(pred_file)
    cands = sorted(Path(save_dir).glob("samples_*.pt"))
    if not cands:
        raise FileNotFoundError(f"No samples_*.pt found in {save_dir}")
    if len(cands) > 1:
        # Prefer the unfiltered sample file written by generate_from_pdb.
        unfiltered = [p for p in cands if "unfiltered" in p.name]
        if unfiltered:
            return unfiltered[0]
    return cands[0]


def ensure_ref_pdb(data_path, ref_pdb):
    """Copy a reference PDB into <data_path>/ref_test_pdbs and return its path.

    evaluate_metrics rewrites every reference PDB that does not live in
    <data_path>/ref_test_pdbs, and asserts that the rewritten file exists.
    """
    src = Path(ref_pdb)
    dst_dir = Path(data_path) / "ref_test_pdbs"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst)


def build_predictions(samples, data_path):
    gen_ligs = samples["gen_ligs"]
    ref_lig = samples.get("ref_lig")
    ref_lig_with_hs = samples.get("ref_lig_with_hs") or ref_lig

    pdb = samples["ref_pdb"]
    pdb_with_hs = samples.get("ref_pdb_with_hs") or pdb
    ref_pdb = ensure_ref_pdb(data_path, pdb)
    ref_pdb_with_hs = ensure_ref_pdb(data_path, pdb_with_hs)

    return [
        {
            "gen_ligs": [gen_ligs],
            "native_ligs": [ref_lig],
            "native_ligs_with_hs": [ref_lig_with_hs],
            "ref_pdbs": [ref_pdb],
            "ref_pdbs_with_hs": [ref_pdb_with_hs],
        }
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--pred_file", default=None)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset", default="spindr")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--return_interaction_list", action="store_true")
    parser.add_argument("--valid_unique", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    samples_path = find_samples(args.pred_file, save_dir)
    print(f"Evaluating {samples_path}")
    samples = torch.load(samples_path, map_location="cpu")

    predictions = build_predictions(samples, args.data_path)
    torch.save(predictions, save_dir / "predictions.pt")

    metric_args = SimpleNamespace(
        num_workers=args.num_workers,
        data_path=args.data_path,
        dataset=args.dataset,
        save_dir=str(save_dir),
        multiple_files=False,
        valid_unique=args.valid_unique,
        remove_hs=args.remove_hs,
        return_interaction_list=args.return_interaction_list,
        evaluate_pilot=False,
    )
    evaluate_metrics.main(metric_args)

    metrics_file = save_dir / "metrics.pt"
    if metrics_file.exists():
        shutil.copy2(metrics_file, save_dir / "metrics_gpr119.pt")
        print(f"Saved {save_dir / 'metrics_gpr119.pt'}")


if __name__ == "__main__":
    main()
