import argparse
import os
import pickle
import time
import warnings
from pathlib import Path

import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pymol2
import torch
from rdkit import Chem
from tqdm import tqdm

import flowr.scriptutil as util
import flowr.util.rdkit as smolRD
from flowr.eval.evaluate_metrics import (
    calc_gb3_metrics_parallel,
    calc_metrics,
    compute_sbdd_metrics_parallel,
)
from flowr.util.metrics import (
    evaluate_uniqueness,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_rdkit_mol(mol, sanitize=False, remove_hs=False):
    coords = mol["coords"].numpy()
    atomics = mol["atomics"].numpy()
    bond_types = mol["bond_types"]
    bond_indices = mol["bond_indices"]
    bonds = torch.cat((bond_indices, bond_types.unsqueeze(1)), dim=-1).numpy()
    charges = mol["charges"].numpy()
    atomics = atomics.tolist()
    tokens = [smolRD.PT.symbol_from_atomic(a) for a in atomics]
    mol = smolRD.mol_from_atoms(
        coords, tokens, bonds, charges, sanitise=False, kekulize=False
    )
    if remove_hs:
        mol = Chem.RemoveHs(
            mol
        )  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def write_sdf_file(sdf_path, molecules, extract_mol=False):
    w = Chem.SDWriter(str(sdf_path))
    for m in molecules:
        if extract_mol:
            if m.rdkit_mol is not None:
                w.write(m.rdkit_mol)
        else:
            if m is not None:
                w.write(m)
    w.close()


def args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data-path",
        type=str,
        help="Path to the data directory",
        required=True,
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="plinder",
        help="Dataset name",
    )
    argparser.add_argument(
        "--save-dir",
        type=str,
        help="Path to the save directory",
        required=True,
    )
    argparser.add_argument(
        "--state",
        type=str,
        help="train or test",
        required=True,
    )
    argparser.add_argument(
        "--remove-hs",
        action="store_true",
        help="Remove hydrogens from the ligands",
    )
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=24,
        help="Number of workers for parallel processing",
        required=True,
    )
    argparser.add_argument(
        "--return-list",
        action="store_true",
        help="Return list of metrics",
    )
    args = argparser.parse_args()
    return args


def load_data(data_path: str, remove_hs: bool, state: str):
    data = Path(data_path) / f"{state}.smol"
    pdb_files = []
    ligands = []
    ligands_with_hs = []
    bytes_data = data.read_bytes()
    for i, mol_bytes in enumerate(
        tqdm(pickle.loads(bytes_data), desc="Loading molecules")
    ):
        obj = pickle.loads(mol_bytes)
        meta_data = obj["metadata"]

        holo_obj = pickle.loads(obj["holo"])
        holo_atoms = holo_obj["atoms"]
        holo_bonds = holo_obj["bonds"]
        holo_atoms.bonds = holo_bonds

        if len(holo_atoms.chain_id[0]) > 1:
            holo_atoms.chain_id = np.array([id[-1] for id in holo_atoms.chain_id])

        # set paths
        pdb_path_with_hs = Path(f"{data_path}/{state}_pdbs_with_hs")
        cif_path_with_hs = Path(f"{data_path}/{state}_cifs_with_hs")
        sdf_path_with_hs = Path(f"{data_path}/{state}_sdfs_with_hs")
        pdb_path_with_hs.mkdir(parents=True, exist_ok=True)
        cif_path_with_hs.mkdir(parents=True, exist_ok=True)
        sdf_path_with_hs.mkdir(parents=True, exist_ok=True)

        try:
            pdb_file = pdb.PDBFile()
            pdb.set_structure(pdb_file, holo_atoms)
            pdb_p = pdb_path_with_hs / f"{meta_data['system_id']}_with_hs.pdb"
            pdb_file.write(Path(pdb_p))
            pdb_files.append(pdb_p)
        except Exception:
            # pocket with Hs
            cif_file = pdbx.CIFFile()
            pdbx.set_structure(cif_file, holo_atoms)
            cif_p = cif_path_with_hs / f"{meta_data['system_id']}_with_hs.cif"
            cif_file.write(cif_p)
            with pymol2.PyMOL() as pymol:
                pymol.cmd.load(cif_p, "holo")
                pdb_file = Path(str(cif_p).replace(".cif", ".pdb"))
                pdb_p = pdb_path_with_hs / pdb_file.name
                pymol.cmd.save(str(pdb_p), selection="holo")
                pdb_files.append(pdb_p)

        ### ligands
        ligand_obj = pickle.loads(obj["ligand"])
        # ref ligands with Hs
        ligand = get_rdkit_mol(ligand_obj, sanitize=True, remove_hs=remove_hs)
        ligand_with_hs = get_rdkit_mol(ligand_obj, sanitize=True, remove_hs=False)
        ligands.append(ligand)
        ligands_with_hs.append(ligand_with_hs)
    return ligands, ligands_with_hs, pdb_files


def main(args):

    ligands, ligands_with_hs, pdb_files = load_data(
        data_path=args.data_path, remove_hs=args.remove_hs, state=args.state
    )

    start_time = time.time()
    print(f"Calculating metrics for {len(ligands)} {args.state} molecules...")

    # Uniqueness
    uniqueness = evaluate_uniqueness(ligands)
    uniqueness = {"Uniqueness": uniqueness}

    # Other molecular metrics
    results_mol = calc_metrics(args, ligands)
    print(
        f"Molecule metrics computed in {round((time.time() - start_time) / 60, 2)} minutes."
    )
    # GenBench3D
    ligands = [[lig] for lig in ligands]
    gb3_metrics = calc_gb3_metrics_parallel(
        ligands,
        num_workers=min(args.num_workers, len(pdb_files)),
    )
    results_mol = {**results_mol, **uniqueness, **gb3_metrics}

    # SBDD metrics
    start_time = time.time()
    # if args.state == "train":
    print(f"Computing SBDD metrics on {args.num_workers} CPUs...")
    results_sbdd = compute_sbdd_metrics_parallel(
        ligands,
        ligands_with_hs,
        pdb_files,
        num_workers=min(args.num_workers, len(pdb_files)),
    )
    print(
        f"SBDD metrics computed in {round((time.time() - start_time) / 60, 2)} minutes."
    )
    results = {**results_mol, **results_sbdd}

    if args.remove_hs:
        args.state += "_no_hs"
    torch.save(results, Path(args.save_dir) / f"metrics_{args.state}.pt")
    util.print_results(results)

    print("Evaluation complete. Script finished.")


if __name__ == "__main__":
    args = args()
    main(args)
