import argparse
import os
import subprocess
import tempfile
from glob import glob
from itertools import zip_longest
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import hydride
import MDAnalysis as mda
import numpy as np
import prolif as plf
from Bio.PDB.Polypeptide import is_aa
from rdkit import Chem
from tqdm import tqdm

from flowr.util.molrepr import GeometricMol
from flowr.util.pocket import (
    PROLIF_INTERACTIONS,
    BindingInteractions,
    PocketComplex,
    PocketComplexBatch,
    ProteinPocket,
)
from posecheck.utils.biopython import (
    ids_scriptly_increasing,
    load_biopython_structure,
    remove_connect_lines,
    reorder_ids,
    save_biopython_structure,
)

"""
This script is used to preprocess data that comes as PDB (protein), SDF (ligand) and optionally txt files (residues in a given radius).
"""


def save_systems_(args, systems, split):
    batch = PocketComplexBatch(systems)
    save_dir = Path(args.save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = save_dir / f"{split}.smol"
    bytes_data = batch.to_bytes()
    save_file.write_bytes(bytes_data)
    print(f"Saved {len(systems)} systems to {save_file.resolve()}")


def process_pdb(
    pdb_file,
    txt_path=None,
    ligand=None,
    add_bonds_to_protein=True,
    add_hs_to_protein=False,
    pocket_cutoff=6.0,
    cut_pocket=False,
    pocket_size_threshold=None,
):

    if cut_pocket:
        assert ligand is not None, "Ligand must be provided to cut out pocket"

    # Load protein from PDB
    if str(pdb_file).endswith(".cif"):
        pdb_file = pdbx.CIFFile.read(str(pdb_file))
        read_fn = pdbx.get_structure
    else:
        pdb_file = pdb.PDBFile.read(str(pdb_file))
        read_fn = pdb.get_structure

    # Sometimes reading charges fails so try to read first but fall back to without charge if not
    try:
        extra = ["charge"]
        structure = read_fn(
            pdb_file, model=1, extra_fields=extra, include_bonds=add_bonds_to_protein
        )
    except Exception:
        structure = read_fn(pdb_file, model=1, include_bonds=add_bonds_to_protein)

    if add_hs_to_protein:
        structure, mask = hydride.add_hydrogen(structure)
        structure.coord = hydride.relax_hydrogen(structure)
    if structure.bonds is None:
        bonds = struc.connect_via_residue_names(structure, inter_residue=True)
        structure.bonds = bonds
        if structure.bonds.as_array().shape[0] == 0:
            raise ValueError("No bonds found in the structure")

    if txt_path is not None:
        with open(txt_path, "r") as f:
            ids = f.read().split()
            try:
                holo_residue_ids = [int(res.split(":")[-1]) for res in ids]
            except ValueError:
                try:
                    holo_residue_ids = [int(res.split(":")[0]) for res in ids]
                except ValueError:
                    try:
                        holo_residue_ids = [int(res.split(".")[0]) for res in ids]
                    except ValueError:
                        holo_residue_ids = [int(res.split(".")[-1]) for res in ids]

            # holo_chain_id = ids[0].split(":")[0]
        structure = structure[np.isin(structure.res_id, holo_residue_ids)]

    if cut_pocket:
        # Cut pocket
        res_ids = set(structure.res_id)
        res_id_filter = []
        for res_id in res_ids:
            res = structure[structure.res_id == res_id]
            if (
                is_aa(res.res_name[0], standard=True)
                and (
                    np.linalg.norm(
                        res.coord[:, None, :] - np.array(ligand._coords[None, :, :]),
                        axis=-1,
                    )
                ).min()
                < pocket_cutoff
            ):
                res_id_filter.append(res_id)
        structure = structure[np.isin(structure.res_id, res_id_filter)]

    pocket = ProteinPocket.from_pocket_atoms(structure)
    return pocket


def load_protein_prolif(protein_path: str):
    """Load protein from PDB file using MDAnalysis
    and convert to plf.Molecule. Assumes hydrogens are present."""
    prot = mda.Universe(protein_path)
    prot = plf.Molecule.from_mda(prot, NoImplicit=False)
    return prot


def load_protein_from_pdb(pdb_path: str, add_hs: bool = False):
    """Load protein from PDB file, add hydrogens, and convert it to a prolif.Molecule.

    Args:
        pdb_path (str): The path to the PDB file.

    Returns:
        plf.Molecule: The loaded protein as a prolif.Molecule.
    """

    tmp_path = tempfile.mkstemp()[1] + ".pdb"
    tmp_protonated_path = tempfile.mkstemp()[1] + ".pdb"

    # Reorder residue IDs if necessary
    structure = load_biopython_structure(pdb_path)
    if not ids_scriptly_increasing(structure):
        structure = reorder_ids(structure)
    save_biopython_structure(structure, tmp_path)  # Save reordered structure

    if add_hs:
        # Run Hydrite
        cmd = f"hydride -i {tmp_path} -o {tmp_protonated_path}"
        out = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Check if Hydrite failed
        if out.returncode != 0:
            print(out.stdout.decode())
            print(out.stderr.decode())
            raise Exception("Hydrite failed")

        # - Remove CONECT lines from the PDB file - #
        # This is necessary because the CONECT lines are not handled correctly by MDAnalysis
        # and they are for some reason added by Hydride
        remove_connect_lines(tmp_protonated_path)
    else:
        tmp_protonated_path = tmp_path

    # Load the protein from the temporary PDB file
    prot = load_protein_prolif(tmp_protonated_path)
    os.remove(tmp_protonated_path)

    return prot


def process_complex(
    struct_path,
    sdf_path=None,
    txt_path=None,
    remove_hs=False,
    kekulize=False,
    add_bonds_to_protein=True,
    add_hs_to_protein=False,
    pocket_cutoff=6.0,
    cut_pocket=False,
    pocket_size_threshold=None,
    compute_interactions=False,
    pocket_type="holo",
    split=None,
):
    ligand = None
    if sdf_path is not None:
        if sdf_path.endswith(".sdf"):
            mol = Chem.SDMolSupplier(str(sdf_path), removeHs=remove_hs)[0]
            if mol is None:
                mol = Chem.MolFromMolFile(str(sdf_path), removeHs=remove_hs)
        elif sdf_path.endswith(".pdb"):
            mol = Chem.MolFromPDBFile(str(sdf_path), removeHs=remove_hs)
        else:
            raise ValueError("Ligand file must be in SDF or PDB format")
        if mol is None:
            print(f"Could not read ligand from {sdf_path}. Skipping.")
            return
        try:
            ligand = GeometricMol.from_rdkit(mol, kekulize=kekulize)
        except Exception as e:
            print(f"Error processing ligand {sdf_path}: {e}")
            return
        ligand.full_mol = ligand.copy()

    # Load protein from PDB
    pocket = process_pdb(
        struct_path,
        ligand=ligand,
        txt_path=txt_path,
        add_bonds_to_protein=add_bonds_to_protein,
        add_hs_to_protein=add_hs_to_protein,
        pocket_cutoff=pocket_cutoff,
        cut_pocket=cut_pocket,
        pocket_size_threshold=pocket_size_threshold,
    )
    pocket.full_pocket = pocket.copy()

    if len(pocket) < 10:
        print(f"Too small or empty pocket after processing {struct_path}. Skipping!")
        return
    if pocket_size_threshold is not None:
        # Check if the pocket size is below the threshold
        if len(pocket) > pocket_size_threshold:
            print(
                f"Pocket size of {len(pocket)} is above threshold {pocket_size_threshold} for structure {struct_path}. Skipping!"
            )
            return

    # get metadata
    pdb_id = Path(struct_path).stem
    metadata = {
        "system_id": pdb_id,
        "is_covalent": False,
    }
    metadata["apo_type"] = None
    metadata["split"] = split

    if pocket_type == "holo":
        _complex = PocketComplex(
            holo=pocket,
            ligand=ligand,
            apo=None,
            metadata=metadata,
        )
    elif pocket_type == "apo":
        _complex = PocketComplex(
            holo=None,
            ligand=ligand,
            apo=pocket,
            metadata=metadata,
        )
    else:
        raise ValueError(f"Invalid pocket type: {pocket_type}")
    _complex.store_metrics_()
    if compute_interactions:
        interactions = BindingInteractions.from_system(
            _complex, interaction_types=PROLIF_INTERACTIONS
        )
        _complex.interactions = interactions.array
    return _complex


def pdb_to_sdf(pdb_path: str, sdf_path: str):
    """Convert a PDB file to an SDF file using Open Babel.

    Args:
        pdb_path (str): The path to the PDB file.
        sdf_path (str): The path to the output SDF file.
    """

    # Convert the PDB file to an SDF file using Open Babel
    cmd = f"obabel -ipdb {pdb_path} -osdf -O {sdf_path}"
    out = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Check if Open Babel failed
    if out.returncode != 0:
        print(out.stdout.decode())
        print(out.stderr.decode())
        raise Exception("Open Babel failed")

    return sdf_path


def main(args):

    data = os.path.join(args.data_path, args.split)
    txt_files = glob(os.path.join(data, "*.txt"))
    pdb_files = [
        os.path.join(Path(file).parent, Path(file).stem.split("_")[0] + ".pdb")
        for file in txt_files
    ]
    if "pocket_ids" in txt_files[0]:
        sdf_files = [
            os.path.join(
                Path(file).parent,
                Path(file).name.replace(
                    "pocket_ids.txt", f"{Path(file).stem.split('_')[0]}.sdf"
                ),
            )
            for file in txt_files
        ]
    else:
        sdf_files = [
            os.path.join(Path(file).parent, Path(file).stem + ".sdf")
            for file in txt_files
        ]

    systems = []
    for txt_path, struct_path, sdf_path in tqdm(
        zip_longest(txt_files, pdb_files, sdf_files, fillvalue=None),
        total=len(pdb_files),
    ):

        # Load ligand from SDF
        # ligand_structure = biotite.structure.io.load_structure(sdf_path)
        # ligand_structure.res_name = ["LIG"] * len(ligand_structure)
        _complex = process_complex(
            struct_path,
            sdf_path=sdf_path,
            txt_path=txt_path,
            remove_hs=args.remove_hs,
            kekulize=args.kekulize,
            add_bonds_to_protein=args.add_bonds_to_protein,
            add_hs_to_protein=args.add_hs_to_protein,
            pocket_cutoff=args.pocket_cutoff,
            cut_pocket=args.cut_pocket,
            compute_interactions=args.compute_interactions,
            split=args.split,
        )

        systems.append(_complex)

    save_systems_(args, systems, args.split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Preprocess data that comes as PDB (protein), SDF (ligand) and TXT (residues in a given radius) files"
    )

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--add_bonds_to_protein", action="store_true")
    parser.add_argument("--add_hs_to_protein", action="store_true")
    parser.add_argument("--pocket_cutoff", type=float, default=6.0)
    parser.add_argument("--cut_pocket", action="store_true")
    parser.add_argument("--compute_interactions", action="store_true")
    parser.add_argument("--kekulize", action="store_true")

    args = parser.parse_args()
    main(args)
