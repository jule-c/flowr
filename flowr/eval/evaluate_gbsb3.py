import argparse
import os
from pathlib import Path

import torch
import yaml

from flowr.eval.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.util.metrics import evaluate_gbsb3_metrics
from flowr.util.rdkit import sanitize_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    args = parser.parse_args()

    # Load the data
    if args.evaluate_pilot:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )
    else:
        gen_ligs, _, ref_ligs, _, ref_pdbs = gather_predictions(
            args, multiple_files=args.multiple_files
        )
    ref_pdbs = [
        os.path.join(
            args.data_path, "test", Path(path).stem.split("_with_hs")[0] + ".pdb"
        )
        for path in ref_pdbs
    ]
    gen_ligs_san = [sanitize_list(mols, filter_uniqueness=False) for mols in gen_ligs]
    config = yaml.safe_load(open("./genbench3d/config/default.yaml", "r"))
    gb3sb_metrics = evaluate_gbsb3_metrics(
        gen_ligs_san,
        ref_ligs=ref_ligs,
        pdb_files=ref_pdbs,
        config=config,
        minimize=False,
        return_dict=True,
    )
    # Save the results
    save_path = os.path.join(args.save_dir, "gbsb3_metrics_dict.pt")
    torch.save(gb3sb_metrics, save_path)
    print(f"GenBench3D metrics results saved to {save_path}")

    import pdb

    pdb.set_trace()
