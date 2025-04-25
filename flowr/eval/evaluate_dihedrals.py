import argparse
import itertools
import time

import numpy as np
import torch
from rdkit.Chem import rdchem, rdMolTransforms
from scipy.stats import wasserstein_distance

from flowr.eval.evaluate_dataset import load_data
from flowr.eval.evaluate_util import gather_predictions, gather_predictions_pilot
from flowr.util.rdkit import sanitize_list


def calc_dihedrals(molecules):
    """
    Calculates dihedral angles (in degrees) for all rotatable bonds in a list of RDKit molecules.
    Assumes each molecule has at least one conformer with valid 3D coordinates.
    """
    dihedrals = []
    for mol in molecules:
        try:
            conf = mol.GetConformer()
        except ValueError:
            continue
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            if bond.GetBondType() != rdchem.BondType.SINGLE:
                continue

            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            if begin_atom.GetDegree() < 2 or end_atom.GetDegree() < 2:
                continue

            begin_neighbors = [
                nbr.GetIdx()
                for nbr in begin_atom.GetNeighbors()
                if nbr.GetIdx() != end_atom.GetIdx()
            ]
            end_neighbors = [
                nbr.GetIdx()
                for nbr in end_atom.GetNeighbors()
                if nbr.GetIdx() != begin_atom.GetIdx()
            ]

            if not begin_neighbors or not end_neighbors:
                continue

            idx1 = begin_neighbors[0]
            idx2 = begin_atom.GetIdx()
            idx3 = end_atom.GetIdx()
            idx4 = end_neighbors[0]

            try:
                angle = rdMolTransforms.GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
                dihedrals.append(angle)
            except Exception:
                continue
    return dihedrals


def calc_wasserstein_distance(molecules1, molecules2):
    """
    Calculates the Wasserstein distance between two sets of dihedral angles (raw values)
    computed from two lists of RDKit molecules.

    Args:
        molecules1 (list): List of RDKit molecule objects for distribution 1.
        molecules2 (list): List of RDKit molecule objects for distribution 2.

    Returns:
        float: The Wasserstein distance between the two distributions of dihedral angles.
    """
    angles1 = calc_dihedrals(molecules1)
    angles2 = calc_dihedrals(molecules2)

    # Compute the 1D Wasserstein distance using SciPy
    return wasserstein_distance(angles1, angles2)


def calc_circular_wasserstein_distance(molecules1, molecules2, nbins=36):
    """
    Calculates the circular Wasserstein distance between two dihedral angle distributions.

    The method bins the dihedral angles over the range [-180, 180] and computes the
    cumulative distribution functions (CDFs) for both. The optimal alignment over the circle
    is achieved by subtracting the median difference between the CDFs. The distance is then
    computed as the sum of absolute differences (multiplied by the bin width).

    Args:
        molecules1 (list): List of RDKit molecule objects for distribution 1.
        molecules2 (list): List of RDKit molecule objects for distribution 2.
        nbins (int): Number of bins to use for the histograms (default: 36, i.e., 10° per bin).

    Returns:
        float: The circular Wasserstein distance between the two distributions.
    """
    # Compute dihedral angles for each set of molecules
    angles1 = np.array(calc_dihedrals(molecules1))
    angles2 = np.array(calc_dihedrals(molecules2))

    # Define histogram bins for the periodic angles ([-180, 180])
    bins = np.linspace(-180, 180, nbins + 1)
    bin_width = bins[1] - bins[0]

    # Compute histograms (counts)
    hist1, _ = np.histogram(angles1, bins=bins)
    hist2, _ = np.histogram(angles2, bins=bins)

    # Normalize histograms to obtain probability mass functions.
    if hist1.sum() > 0:
        p1 = hist1.astype(float) / hist1.sum()
    else:
        p1 = np.zeros_like(hist1)

    if hist2.sum() > 0:
        p2 = hist2.astype(float) / hist2.sum()
    else:
        p2 = np.zeros_like(hist2)

    # Compute cumulative distributions (CDFs)
    cdf1 = np.cumsum(p1)
    cdf2 = np.cumsum(p2)

    # Calculate the difference between the CDFs
    diff = cdf1 - cdf2
    # The optimal circular alignment is found by subtracting the median difference
    shift = np.median(diff)
    # Circular Wasserstein distance is computed as the total absolute deviation, scaled by bin width
    distance = np.sum(np.abs(diff - shift)) * bin_width
    return distance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--valid_unique", action="store_true")
    parser.add_argument("--return_list", action="store_true")
    parser.add_argument("--multiple_files", action="store_true")
    parser.add_argument("--evaluate_pilot", action="store_true")
    parser.add_argument(
        "--compare_to", type=str, default="test", choices=["train", "test", "val"]
    )
    parser.add_argument(
        "--circular", action="store_true", help="Use circular Wasserstein distance"
    )
    args = parser.parse_args()

    # Load the ground truth data
    ligs, _, _ = load_data(
        data_path=args.data_path, remove_hs=True, state=args.compare_to
    )

    # Load the generated data
    if args.evaluate_pilot:
        gen_ligs, _, _, _, _ = gather_predictions_pilot(
            args, multiple_files=args.multiple_files
        )
    else:
        gen_ligs, _, _, _, _ = gather_predictions(
            args, multiple_files=args.multiple_files
        )

    start_time = time.time()
    print(
        f"Computing dihedral angles distribution distance for {len(list(itertools.chain(*gen_ligs)))} generated molecules..."
    )

    # Compute the metrics
    gen_ligs_san = [
        sanitize_list(mols, filter_uniqueness=False, sanitize=True) for mols in gen_ligs
    ]
    gen_ligs_san = list(itertools.chain(*gen_ligs_san))
    if args.circular:
        distances = calc_circular_wasserstein_distance(ligs, gen_ligs_san)
    else:
        distances = calc_wasserstein_distance(ligs, gen_ligs_san)

    # Save the results
    save_path = f"{args.save_dir}/dihedral_angles_distance_to_{args.compare_to}_set.pt"
    torch.save(distances, save_path)
    print(f"Results saved to {save_path}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Average distance to {args.compare_to}: {distances}")
