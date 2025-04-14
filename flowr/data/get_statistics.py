import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from flowr.data.data_info import atom_encoder
from flowr.data.util import compute_all_statistics, mol_to_dict
from flowr.util.molrepr import GeometricMol


def save_pickle(array, path, exist_ok=True):
    if exist_ok:
        with open(path, "wb") as f:
            pickle.dump(array, f)
    else:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(array, f)


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the data directory",
        required=True,
    )
    argparser.add_argument(
        "--state",
        type=str,
        help="train, val or test",
        required=True,
    )
    argparser.add_argument(
        "--remove_hs",
        action="store_true",
        help="Remove hydrogens from the molecule",
    )
    args = argparser.parse_args()
    return args


def get_statistics(args):
    data_path = Path(f"{args.data_dir}/{args.state}.smol")
    bytes_data = data_path.read_bytes()
    data_list = []
    failed = []
    for mol_bytes in tqdm(pickle.loads(bytes_data)):
        obj = pickle.loads(mol_bytes)
        ligand = GeometricMol.from_bytes(obj["ligand"])
        meta_data = obj["metadata"]
        if ligand is None:
            failed.append(meta_data["system_id"])
            continue
        data_list.append(
            mol_to_dict(
                ligand.to_rdkit(),
                atom_encoder=atom_encoder,
                remove_hs=args.remove_hs,
            )
        )

    print(f"Number of processed molecules: {len(data_list)}")
    print(f"Number of failed molecules: {failed}")
    print("Computing statistics...")
    statistics = compute_all_statistics(
        data_list,
        atom_encoder,
        charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
        additional_feats=True,
    )

    h = "noh" if args.remove_hs else "h"
    processed_paths = [
        f"{args.state}_{h}.pt",
        f"{args.state}_n_{h}.pickle",
        f"{args.state}_atom_types_{h}.npy",
        f"{args.state}_bond_types_{h}.npy",
        f"{args.state}_charges_{h}.npy",
        f"{args.state}_valency_{h}.pickle",
        f"{args.state}_bond_lengths_{h}.pickle",
        f"{args.state}_angles_{h}.npy",
        f"{args.state}_is_aromatic_{h}.npy",
        f"{args.state}_is_in_ring_{h}.npy",
        f"{args.state}_hybridization_{h}.npy",
        f"{args.state}_is_h_donor_{h}.npy",
        f"{args.state}_is_h_acceptor_{h}.npy",
        f"{args.state}_dihedrals_{h}.npy",
    ]
    if not Path(f"{args.data_dir}/processed").exists():
        os.makedirs(f"{args.data_dir}/processed")
    processed_paths = [f"{args.data_dir}/processed/{p}" for p in processed_paths]
    save_pickle(statistics.num_nodes, processed_paths[1])
    np.save(processed_paths[2], statistics.atom_types)
    np.save(processed_paths[3], statistics.bond_types)
    np.save(processed_paths[4], statistics.charge_types)
    save_pickle(statistics.valencies, processed_paths[5])
    save_pickle(statistics.bond_lengths, processed_paths[6])
    np.save(processed_paths[7], statistics.bond_angles)
    np.save(processed_paths[8], statistics.is_aromatic)
    np.save(processed_paths[9], statistics.is_in_ring)
    np.save(processed_paths[10], statistics.hybridization)
    np.save(processed_paths[11], statistics.is_h_donor)
    np.save(processed_paths[12], statistics.is_h_acceptor)
    np.save(processed_paths[13], statistics.dihedrals)
    print("Statistics computed and saved.")


if __name__ == "__main__":
    args = args()
    get_statistics(args)
