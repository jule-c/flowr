import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch


def create_list_defaultdict():
    return defaultdict(list)


def aggregate_results(save_dir, return_list=False, pattern="energy_metrics_dict_*.pt"):
    import glob

    # Find all files matching the pattern
    files = glob.glob(os.path.join(save_dir, pattern))
    if not files:
        raise FileNotFoundError(
            f"No result files found for pattern {pattern} in {save_dir}"
        )

    # Dictionary to collect all metrics per key across jobs
    aggregated = (
        defaultdict(list) if return_list else defaultdict(create_list_defaultdict)
    )

    for f in files:
        result = torch.load(f)
        # result can be a dict of metrics (if return_list is False)
        for target, value in result.items():
            if isinstance(value, dict):
                for key, val in value.items():
                    assert isinstance(
                        val, (float, int)
                    ), f"Value {val} is not a float or int."
                    aggregated[target][key] = val
            elif isinstance(value, list):
                aggregated[target].extend(value)

    return aggregated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--return_list", action="store_true", help="Return list of values"
    )
    parser.add_argument(
        "--calc_relax_energy",
        action="store_true",
    )
    parser.add_argument(
        "--evaluate_test_data",
        action="store_true",
        help="Evaluate test data",
    )
    parser.add_argument(
        "--evaluate_val_data",
        action="store_true",
        help="Evaluate val data",
    )
    parser.add_argument(
        "--evaluate_train_data",
        action="store_true",
        help="Evaluate train data",
    )
    args = parser.parse_args()

    # Aggregate results
    state = (
        "_test"
        if args.evaluate_test_data
        else (
            "_val"
            if args.evaluate_val_data
            else "_train" if args.evaluate_train_data else ""
        )
    )
    pattern = (
        f"energy_metrics_dict{state}_*.pt"
        if not args.calc_relax_energy
        else f"relax_energy_metrics_dict{state}_*.pt"
    )
    aggregated_results = aggregate_results(
        args.save_dir, return_list=args.return_list, pattern=pattern
    )

    # Save the results
    if args.evaluate_test_data or args.evaluate_val_data or args.evaluate_train_data:
        if args.calc_relax_energy:
            save_path = os.path.join(
                args.save_dir, f"relax_energy_metrics_dict{state}.pt"
            )
        else:
            save_path = os.path.join(args.save_dir, f"energy_metrics_dict{state}.pt")
    else:
        if args.calc_relax_energy:
            save_path = os.path.join(args.save_dir, "relax_energy_metrics_dict.pt")
        else:
            save_path = os.path.join(args.save_dir, "energy_metrics_dict.pt")
    torch.save(aggregated_results, save_path)
    print(f"Aggregated results saved to {save_path}")

    # print mean results
    units = "Eh" if not args.calc_relax_energy else "kcal/mol"
    mean_results = []
    std_results = []
    num_targets = len(aggregated_results)
    if args.return_list:
        if (
            args.evaluate_test_data
            or args.evaluate_val_data
            or args.evaluate_train_data
        ):
            all_ligs = len(aggregated_results.values())
            mean_results = [
                v[0]
                for v in aggregated_results.values()
                if len(v) > 0 and not pd.isna(v[0])
            ]
            std_results = np.nanstd(mean_results)
            success_ligs = len(mean_results)
        else:
            success_ligs = 0
            all_ligs = 0
            for _, values in aggregated_results.items():
                all_ligs += len(values)
                values = [v for v in values if not pd.isna(v)]
                success_ligs += len(values)
                mean_results.append(np.nanmean(values))
                std_results.append(np.nanstd(values))
        print(
            f"Calculated mean and std total energy for {success_ligs} out of {all_ligs} ligands across {num_targets} targets."
        )
        print(
            f"Failed calculations: {round(100 * (all_ligs - success_ligs) / all_ligs, 2)}%"
        )
        print(
            f"Mean Energy ({units}): {np.nanmean(mean_results)} +- {np.nanmean(std_results)}"
        )
    else:
        print(f"Aggregated results: {aggregate_results}")
