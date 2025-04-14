import argparse
import signal
import tempfile
import time
import warnings
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import lightning as L
import numpy as np
import torch
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

import flowr.scriptutil as util
import flowr.util.rdkit as smolRD
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.datasets import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricNoiseSampler,
)
from flowr.data.preprocess_pdbs import process_complex
from flowr.models.fm_pocket import LigandPocketCFM
from flowr.models.pocket import LigandGenerator, PocketEncoder
from flowr.scriptutil import generate_ligands_per_target
from flowr.util.functional import (
    LigandPocketOptimization,
    add_and_optimize_hs,
)
from flowr.util.metrics import interaction_recovery_per_complex
from flowr.util.pocket import PROLIF_INTERACTIONS, PocketComplexBatch
from flowr.util.rdkit import write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Default script arguments
DEFAULT_BUCKET_COST_SCALE = "quadratic"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def write_ligand_pocket_complex_pdb(
    all_gen_ligs, all_gen_pdbs, output_path, complex_name="complex"
):
    """
    Combines ligand and pocket (PDB file) into a ligand-pocket complex and writes them to an SDF file.

    Args:
        all_gen_ligs (list[Chem.Mol]): List of RDKit ligand molecules.
        all_gen_pdbs (list[str]): List of file paths (or paths as strings) for the pocket PDBs.
        output_path (str): File path to write the combined complexes (PDB format).
        complex_name (str): Name prefix for each complex. Default is "Complex".

    Notes:
        This function assumes that the PDB files can be parsed by RDKit via Chem.MolFromPDBFile.
    """
    if not all_gen_ligs:
        raise ValueError("No ligand molecules provided.")
    if not all_gen_pdbs:
        raise ValueError("No pocket PDB files provided.")

    for i, (lig, pdb_file) in enumerate(zip(all_gen_ligs, all_gen_pdbs)):
        out_path = Path(output_path) / f"{complex_name}_{i}.pdb"
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            lig_sdf = tmp.name
            write_sdf_file(lig_sdf, [lig])
            cmd.reinitialize()
            cmd.load(pdb_file, "protein")
            cmd.load(lig_sdf, "ligand")
            cmd.save(out_path, "protein or ligand")
            cmd.reinitialize()


def write_ligand_pocket_complex_sdf(
    all_gen_ligs, all_gen_pdbs, output_path, complex_name="complex"
):
    """
    Combines ligand and pocket (PDB file) into a ligand-pocket complex and writes them to an SDF file.

    Args:
        all_gen_ligs (list[Chem.Mol]): List of RDKit ligand molecules.
        all_gen_pdbs (list[str]): List of file paths (or paths as strings) for the pocket PDBs.
        output_path (str): File path to write the combined complexes (SDF format).
        complex_name (str): Name prefix for each complex. Default is "Complex".

    Notes:
        This function assumes that the PDB files can be parsed by RDKit via Chem.MolFromPDBFile.
    """
    if not all_gen_ligs:
        raise ValueError("No ligand molecules provided.")
    if not all_gen_pdbs:
        raise ValueError("No pocket PDB files provided.")

    complexes = []
    for i, (lig, pdb_file) in enumerate(zip(all_gen_ligs, all_gen_pdbs)):

        # Parse the pocket molecule from the PDB file.
        pocket_mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
        if pocket_mol is None:
            print(
                f"Warning: Could not parse pocket from {pdb_file}; skipping entry {i}."
            )
            continue

        # Combine the pocket and ligand.
        # Note: CombineMols doesn't merge conformers. The output will have separate conformers.
        complex_mol = Chem.CombineMols(pocket_mol, lig)
        complex_mol.SetProp("Name", f"{complex_name}_{i}")
        complexes.append(complex_mol)

    if not complexes:
        raise RuntimeError("No valid complexes were created.")

    # Write complexes to SDF
    sdf_path = Path(output_path) / f"{complex_name}.sdf"
    writer = Chem.SDWriter(sdf_path)
    for mol in complexes:
        writer.write(mol)
    writer.close()
    print(f"Wrote {len(complexes)} ligand-pocket complexes to {output_path}")


def process_lig_wrapper(
    pair, optimizer, optimize_pocket_hs=False, process_pocket=False
):
    """
    Wrapper for process_lig to enable pickling.

    Args:
        pair (tuple): A tuple containing (lig, pdb_file).
        optimizer: The optimizer to pass to process_lig.

    Returns:
        The processed ligand.
    """
    lig, pdb_file = pair
    return process_lig(
        lig,
        pdb_file=pdb_file,
        optimizer=optimizer,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
    )


def process_lig(
    lig, pdb_file, optimizer, optimize_pocket_hs=False, process_pocket=False
):
    """
    Add hydrogens to a ligand and optimize it.
    """
    return add_and_optimize_hs(
        lig,
        pdb_file,
        optimizer=optimizer,
        optimize_pocket_hs=optimize_pocket_hs,
        process_pocket=process_pocket,
    )


def process_interaction(
    gen_lig, ref_lig, pdb_file, pocket_cutoff, save_dir, remove_hs, protonate_ligands
):
    recovery_rate, tanimoto_sim = interaction_recovery_per_complex(
        gen_ligs=gen_lig,
        native_lig=ref_lig,
        pdb_file=pdb_file,
        add_optimize_gen_lig_hs=remove_hs and not protonate_ligands,
        add_optimize_ref_lig_hs=False,
        optimize_pocket_hs=False,
        process_pocket=False,
        optimization_method="prolif_mmff",
        pocket_cutoff=pocket_cutoff,
        strip_invalid=True,
        save_dir=save_dir,
        return_list=False,
    )
    return {
        "recovery_rate": recovery_rate,
        "tanimoto_sim": tanimoto_sim,
    }


def get_fingerprints(
    mols: Iterable[Chem.Mol], radius=2, length=4096, chiral=True, sanitize=False
):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    if sanitize:
        fps = []
        for mol in mols:
            Chem.SanitizeMol(mol)
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, length, useChirality=chiral
                )
            )
    else:
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]
    return fps


def filter_diverse_ligands(ligands, threshold=0.9):
    """
    Filters ligands by diversity based on Tanimoto similarity.
    Molecules with Tanimoto similarity greater than `threshold` to
    any already kept molecule are filtered out.

    Args:
        ligands (list): List of RDKit Chem.Mol objects.
        threshold (float): Tanimoto similarity threshold, default 0.9.

    Returns:
        list: Filtered list of diverse Chem.Mol objects.
    """
    selected_ligs = []
    selected_fps = []
    for lig in ligands:
        # Compute the Morgan fingerprint with radius 2 (default bit vector size)
        fp = AllChem.GetMorganFingerprint(lig, radius=2)
        # Check if fingerprint is too similar to any already selected one.
        if any(
            DataStructs.TanimotoSimilarity(fp, sel_fp) > threshold
            for sel_fp in selected_fps
        ):
            continue
        selected_ligs.append(lig)
        selected_fps.append(fp)
    return selected_ligs


def filter_diverse_ligands_bulk(ligands, pdbs=None, threshold=0.9):
    """
    Filters ligands by diversity using a bulk computation of Tanimoto similarities.
    The function computes Morgan fingerprints for all ligands, builds a symmetric
    pairwise similarity matrix using BulkTanimotoSimilarity (with the diagonal set to zero),
    and then greedily selects a set of ligands such that any ligand added has a Tanimoto similarity
    of at most `threshold` (default 0.9) with every ligand already selected.

    Args:
        ligands (list): List of RDKit Chem.Mol objects.
        threshold (float): Tanimoto similarity threshold. Default: 0.9.

    Returns:
        list: Filtered list of diverse Chem.Mol objects.
    """
    # Compute Morgan fingerprints (with radius 2)
    fps = get_fingerprints(ligands, radius=2, length=4096, chiral=True, sanitize=False)
    n = len(fps)
    if n == 0:
        return []

    # Compute pairwise similarity matrix; initialize with zeros.
    sim_matrix = np.zeros((n, n))
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        sim_matrix[i, :i] = sims
        sim_matrix[:i, i] = sims

    # Greedy selection: add a ligand only if its similarity with all already selected ligands is <= threshold.
    selected_indices = []
    for i in range(n):
        if selected_indices:
            max_sim = np.max(sim_matrix[i, selected_indices])
            if max_sim > threshold:
                continue
        selected_indices.append(i)

    # Return the molecules corresponding to the selected indices.
    ligands = [ligands[i] for i in selected_indices]
    if pdbs is not None:
        pdbs = [pdbs[i] for i in selected_indices]
        return ligands, pdbs
    return ligands


def load_model(args):
    checkpoint = torch.load(args.ckpt_path)
    hparams = dotdict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    hparams["predict_interactions"] = hparams["predict_interactions"]
    hparams["interaction_inpainting"] = args.interaction_inpainting
    hparams["scaffold_inpainting"] = args.scaffold_inpainting
    hparams["func_group_inpainting"] = args.func_group_inpainting
    hparams["linker_inpainting"] = args.linker_inpainting
    hparams["use_lig_pocket_rbf"] = (
        hparams["use_lig_pocket_rbf"] if "use_lig_pocket_rbf" in hparams else False
    )
    hparams["coord_skip_connect"] = (
        hparams["coord_skip_connect"] if "coord_skip_connect" in hparams else True
    )
    hparams["use_rbf"] = hparams["use_rbf"] if "use_rbf" in hparams else False
    hparams["use_sphcs"] = hparams["use_sphcs"] if "use_sphcs" in hparams else False
    hparams["data_path"] = args.data_path
    hparams["save_dir"] = args.save_dir

    # Number of corrector iterations
    if args.corrector_iters > 0:
        assert (
            args.categorical_strategy == "velocity-sample"
        ), "Only velocity sampling supported for corrector iterations."
        hparams["corrector_iters"] = args.corrector_iters

    # print("Building model vocab...")
    # vocab = util.build_vocab(remove_hs=hparams["remove_hs"])
    # vocab_pocket_atoms = util.build_vocab_pocket_atoms()
    # vocab_pocket_res = util.build_vocab_pocket_res()
    # print("Vocab complete.")

    print("Building model vocab...")
    vocab = util._build_vocab()
    vocab_pocket_atoms = util._build_vocab_pocket_atoms()
    vocab_pocket_res = util._build_vocab_pocket_res()
    print("Vocab complete.")

    if hparams["pocket_noise"] in ["fix", "random"]:
        assert (
            args.arch == "pocket"
        ), "Model trained on rigid pocket flow matching. Change arch to pocket."
        assert (
            args.pocket_type == "holo"
        ), "Model trained on rigid pocket flow matching. Change pocket_type to holo."
    if hparams["pocket_noise"] == "apo":
        assert (
            args.arch == "pocket_flex"
        ), "Model trained on apo pocket flow matching. Change arch to pocket_flex."
        assert (
            args.pocket_type == "apo"
        ), "Model trained on apo pocket flow matching. Change pocket_type to apo."

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if hparams["flow_interactions"] or hparams["predict_interactions"]
        else None
    )
    if args.arch == "pocket":
        fixed_equi = hparams["pocket-fixed_equi"]
        pocket_enc = PocketEncoder(
            hparams["pocket-d_equi"],
            hparams["pocket-d_inv"],
            hparams["d_message"],
            hparams["pocket-n_layers"],
            hparams["n_attn_heads"],
            hparams["d_message_ff"],
            hparams["d_edge"],
            vocab_pocket_atoms.size,
            n_bond_types,
            vocab_pocket_res.size,
            fixed_equi=fixed_equi,
        )
        egnn_gen = LigandGenerator(
            hparams["d_equi"],
            hparams["d_inv"],
            hparams["d_message"],
            hparams["n_layers"],
            hparams["n_attn_heads"],
            hparams["d_message_ff"],
            hparams["d_edge"],
            vocab.size,
            n_bond_types,
            predict_interactions=hparams["predict_interactions"],
            flow_interactions=hparams["flow_interactions"],
            n_interaction_types=n_interaction_types,
            use_lig_pocket_rbf=hparams["use_lig_pocket_rbf"],
            n_extra_atom_feats=1,
            self_cond=hparams["self_cond"],
            coord_skip_connect=hparams["coord_skip_connect"],
            pocket_enc=pocket_enc,
        )
    else:
        raise ValueError(f"Unknown architecture {args.arch}")

    CFM = LigandPocketCFM if args.arch == "pocket" else None
    if args.arch == "pocket":
        from flowr.models.fm_pocket import Integrator
    elif args.arch == "pocket_flex":
        from flowr.models.fm_pocket_flex import Integrator

    # Build the integrator
    type_mask_index = None
    bond_mask_index = None
    integrator = Integrator(
        args.integration_steps,
        type_strategy=args.categorical_strategy,
        bond_strategy=args.categorical_strategy,
        pocket_noise=hparams["pocket_noise"],
        coord_noise_std=args.coord_noise_std,
        ligand_only=hparams["ligand_only"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
    )
    # Build the model
    fm_model = CFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams,
    )
    return fm_model, hparams, vocab, vocab_pocket_atoms, vocab_pocket_res


def load_data(args, hparams, vocab):
    # Load the data
    system = process_complex(
        args.pdb_file,
        sdf_path=args.ligand_file,
        add_bonds_to_protein=True,
        add_hs_to_protein=False,
        pocket_cutoff=args.pocket_cutoff,
        cut_pocket=args.cut_pocket,
        txt_path=args.res_txt_file,
        pocket_type=args.pocket_type,
        compute_interactions=args.compute_interactions,
    )
    system = system.remove_hs(include_ligand=hparams["remove_hs"])
    systems = [system]
    data = PocketComplexBatch(systems)

    # Create the transform function
    if hparams["coord_scale"] == 1.0:
        coord_std = 1.0
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(
        util.complex_transform,
        vocab=vocab,
        n_bonds=n_bond_types,
        coord_std=coord_std,
        use_interactions=hparams["flow_interactions"]
        or hparams["predict_interactions"]
        or hparams["interaction_inpainting"],
        pocket_noise=hparams["pocket_noise"],
    )

    # Create the prior sampler
    type_mask_index = None
    bond_mask_index = None
    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-ligand-prior-type-noise"],
        bond_noise=hparams["val-ligand-prior-bond-noise"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )

    # Create the evaluation interpolant
    if args.pocket_time is not None:
        assert (
            args.separate_pocket_interpolation
        ), "Setting a pocket time requires a separate pocket interpolation"
    if args.interaction_time is not None:
        assert (
            args.separate_interaction_interpolation
        ), "Setting an interaction time requires a separate interaction interpolation"

    ## Determine the categorical sampling strategy
    if args.categorical_strategy == "mask":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "uniform-sample":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "prior-sample":
        assert hparams["val-ligand-type-interpolation"] == "unmask"
        categorical_interpolation = "unmask"
    elif args.categorical_strategy == "velocity-sample":
        assert hparams["val-ligand-type-interpolation"] == "sample"
        categorical_interpolation = "sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )
    eval_interpolant = ComplexInterpolant(
        prior_sampler,
        ligand_coord_interpolation="linear",
        ligand_type_interpolation=categorical_interpolation,
        ligand_bond_interpolation=categorical_interpolation,
        ligand_fixed_time=args.ligand_time,
        pocket_fixed_time=args.pocket_time,
        interaction_fixed_time=args.interaction_time,
        rigid_pocket=hparams["pocket_noise"] == "fix",
        separate_pocket_interpolation=args.separate_pocket_interpolation,
        separate_interaction_interpolation=args.separate_interaction_interpolation,
        n_interaction_types=(
            len(PROLIF_INTERACTIONS)
            if hparams["flow_interactions"]
            or hparams["predict_interactions"]
            or hparams["interaction_inpainting"]
            else None
        ),
        flow_interactions=hparams["flow_interactions"],
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        substructure=args.substructure,
        dataset=args.dataset,
        sample_mol_sizes=args.sample_mol_sizes,
        equivariant_ot=args.use_equi_ot,
        vocab=vocab,
        inference=True,
    )
    return data, transform, eval_interpolant


def get_dataset(data, transform, vocab, interpolant, args, hparams):
    dataset = GeometricDataset(data, data_cls=PocketComplexBatch, transform=transform)
    dataset = dataset.sample_n_molecules_per_target(args.sample_n_molecules_per_target)
    if args.gpus > 1:
        dataset = dataset.split(idx=args.mp_index, n_chunks=args.gpus)
    return dataset


def get_dataloader(dataset, vocab, interpolant, args, hparams, iter=0):
    L.seed_everything(args.seed + iter)
    util.disable_lib_stdout()
    util.configure_fs()

    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        val_batch_size=args.batch_cost,
        dataset=args.dataset,
        vocab=vocab,
        remove_hs=hparams["remove_hs"],
        test_interpolant=interpolant,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )

    test_dl = dm.test_dataloader()
    return test_dl


def create_list_defaultdict():
    return defaultdict(list)


def evaluate(args):
    # Set precision
    torch.set_float32_matmul_precision("high")

    # load hyperparameter
    print(f"Using model stored at {args.ckpt_path}")

    print("Loading model...")
    model, hparams, vocab, vocab_pocket_atoms, vocab_pocket_res = load_model(
        args,
    )
    model = model.to("cuda")
    model.eval()
    print("Model complete.")

    # Load the data
    data, transform, interpolant = load_data(args, hparams, vocab)
    dataset = get_dataset(data, transform, vocab, interpolant, args, hparams)

    if args.gpus > 1:
        sample_n_molecules_per_target = dataset.__len__()
    else:
        sample_n_molecules_per_target = args.sample_n_molecules_per_target

    print("\nStarting sampling...\n")
    k = 0
    num_ligands = 0
    all_gen_ligs = []
    all_gen_pdbs = []
    gen_pdbs = None
    start = time.time()
    out_dict = defaultdict(list)
    while num_ligands < sample_n_molecules_per_target and k <= args.max_sample_iter:
        print(
            f"...Sampling iteration {k + 1}...",
            end="\r",
        )
        dataloader = get_dataloader(dataset, vocab, interpolant, args, hparams, iter=k)
        for i, batch in enumerate(tqdm(dataloader, desc="Sampling", leave=False)):
            if args.arch == "pocket_flex":
                gen_ligs, gen_pdbs = generate_ligands_per_target(
                    args, hparams, model, batch, save_traj=False, iter=f"{k}_{i}"
                )
            else:
                gen_ligs = generate_ligands_per_target(
                    args, hparams, model, batch, save_traj=True, iter=f"{k}_{i}"
                )
            if args.filter_valid_unique:
                if gen_pdbs:
                    gen_ligs, gen_pdbs = smolRD.sanitize_list(
                        gen_ligs,
                        pdbs=gen_pdbs,
                        filter_uniqueness=True,
                        filter_pdb=True,
                        sanitize=True,
                    )
                else:
                    gen_ligs = smolRD.sanitize_list(
                        gen_ligs,
                        filter_uniqueness=True,
                        sanitize=True,
                    )
            all_gen_ligs.extend(gen_ligs)
            if gen_pdbs:
                all_gen_pdbs.extend(gen_pdbs)
            num_ligands += len(gen_ligs)
        if args.filter_valid_unique:
            print(
                f"Validity rate: {round(len(all_gen_ligs) / sample_n_molecules_per_target, 2)}"
            )
        k += 1

    run_time = time.time() - start
    if num_ligands == 0:
        raise (
            f"Reached {args.max_sample_iter} sampling iterations, but could not find any ligands."
        )
    elif num_ligands < args.sample_n_molecules_per_target:
        print(
            f"FYI: Reached {args.max_sample_iter} sampling iterations, but could only find {num_ligands} ligands."
        )
    elif num_ligands > args.sample_n_molecules_per_target:
        all_gen_ligs = all_gen_ligs[: args.sample_n_molecules_per_target]
        if all_gen_pdbs:
            all_gen_pdbs = all_gen_pdbs[: args.sample_n_molecules_per_target]

    data = batch[1]
    ref_ligs = model._generate_ligs(
        data, lig_mask=data["lig_mask"].bool(), scale=model.coord_scale
    )[0]
    ref_lig_with_hs = model.retrieve_ligs_with_hs(data, save_idx=0)
    ref_pdb = model.retrieve_pdbs(
        data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )
    ref_pdb_with_hs = model.retrieve_pdbs_with_hs(
        data, save_dir=Path(args.save_dir) / "ref_pdbs", save_idx=0
    )

    out_dict["gen_ligs"] = all_gen_ligs
    if args.arch == "pocket_flex":
        assert len(all_gen_pdbs) == len(all_gen_ligs)
        out_dict["gen_pdbs"] = all_gen_pdbs
    out_dict["ref_lig"] = ref_ligs
    out_dict["ref_lig_with_hs"] = ref_lig_with_hs
    out_dict["ref_pdb"] = ref_pdb
    out_dict["ref_pdb_with_hs"] = ref_pdb_with_hs
    out_dict["run_time"] = run_time

    print(
        f"\n Run time={round(run_time, 2)}s for {len(out_dict['gen_ligs'])} molecules \n"
    )

    # Filter by diversity
    print("Filtering ligands by diversity...")
    if args.arch == "pocket_flex":
        all_gen_ligs, all_gen_pdbs = filter_diverse_ligands_bulk(
            all_gen_ligs, all_gen_pdbs, threshold=0.9
        )
    else:
        all_gen_ligs = filter_diverse_ligands_bulk(all_gen_ligs, threshold=0.9)
    print(
        f"Number of ligands after filtering by diversity: {len(all_gen_ligs)} ligands ({args.sample_n_molecules_per_target - len(all_gen_ligs)} removed)"
    )

    # Protonate generated ligands:
    if args.protonate_ligands:
        assert hparams[
            "remove_hs"
        ], "The model outputs protonated ligands, no need for additional protonation."
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print("Protonating ligands...")
        optimizer = LigandPocketOptimization(
            pocket_cutoff=args.pocket_cutoff, strip_invalid=True
        )
        # Create a partial function binding pdb_file and optimizer
        if args.arch == "pocket_flex":
            process_lig_partial = partial(
                process_lig_wrapper,
                optimizer=optimizer,
                optimize_pocket_hs=True,
                process_pocket=True,
            )
            with Pool(processes=args.num_workers) as pool:
                all_gen_ligs_hs = pool.map(
                    process_lig_partial, zip(all_gen_ligs, all_gen_pdbs)
                )
        else:
            process_lig_partial = partial(
                process_lig,
                pdb_file=args.pdb_file,
                optimizer=optimizer,
                process_pocket=True,
            )
            with Pool(processes=args.num_workers) as pool:
                all_gen_ligs_hs = pool.map(process_lig_partial, all_gen_ligs)

        all_gen_ligs_hs = [Chem.Mol(lig) for lig in all_gen_ligs_hs]
        out_dict["gen_ligs_hs"] = all_gen_ligs_hs
        print("Done!")

    # Save out_dict as pickle file
    target_name = Path(args.pdb_file).stem
    if args.filter_valid_unique:
        predictions = Path(args.save_dir) / f"samples_{target_name}.pt"
    else:
        predictions = Path(args.save_dir) / f"samples_unfiltered_{target_name}.pt"
    torch.save(out_dict, str(predictions))
    # Save ligands as SDF
    sdf_dir = Path(args.save_dir) / f"samples_{target_name}.sdf"
    write_sdf_file(sdf_dir, all_gen_ligs, name=target_name)
    if args.protonate_ligands:
        sdf_dir = Path(args.save_dir) / f"samples_{target_name}_protonated.sdf"
        write_sdf_file(sdf_dir, all_gen_ligs_hs, name=target_name)

    # Save ligand-pocket complexes
    if args.arch == "pocket_flex":
        gen_complexes_dir = Path(args.save_dir) / "gen_complexes_protonated"
        if not gen_complexes_dir.exists():
            gen_complexes_dir.mkdir(parents=True, exist_ok=True)
        if args.protonate_ligands:
            write_ligand_pocket_complex_pdb(
                all_gen_ligs_hs,
                all_gen_pdbs,
                gen_complexes_dir,
                complex_name=target_name,
            )
        write_ligand_pocket_complex_pdb(
            all_gen_ligs,
            all_gen_pdbs,
            gen_complexes_dir,
            complex_name=target_name,
        )

    print(f"Samples saved to {str(args.save_dir)}")
    print("Sampling finished.")

    if args.compute_interaction_recovery:
        print("Computing interaction recovery...")
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        # Split the list into chunks of size args.num_workers
        chunks = [
            all_gen_ligs_hs[i : i + args.num_workers]
            for i in range(0, len(all_gen_ligs_hs), args.num_workers)
        ]
        process_interaction_partial = partial(
            process_interaction,
            ref_lig=ref_lig_with_hs,
            pdb_file=ref_pdb_with_hs,
            pocket_cutoff=args.pocket_cutoff,
            save_dir=args.save_dir,
            remove_hs=hparams["remove_hs"],
            protonate_ligands=args.protonate_ligands,
        )
        recovery_rates = []
        tanimoto_sims = []
        with Pool(processes=args.num_workers) as pool:
            for chunk in chunks:
                # Process each chunk in parallel.
                chunk_results = pool.map(process_interaction_partial, chunk)
                recovery_rates.extend([res["recovery_rate"] for res in chunk_results])
                tanimoto_sims.extend([res["tanimoto_sim"] for res in chunk_results])

        recovery_rates = [result for result in recovery_rates if result is not None]
        tanimoto_sims = [result for result in tanimoto_sims if result is not None]
        out_dict["interaction_recovery"] = recovery_rates
        out_dict["tanimoto_sims"] = tanimoto_sims
        print(f"Interaction recovery rate: {np.nanmean(recovery_rates)}")
        print(f"Interaction Tanimoto similarity: {np.nanmean(tanimoto_sims)}")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--pdb_file', type=str, required=True)
    parser.add_argument('--ligand_file', type=str, required=True)
    parser.add_argument('--res_txt_file', type=str, default=None)
    
    parser.add_argument('--cut_pocket', action='store_true')
    parser.add_argument('--pocket_cutoff', type=float, default=6.0)
    parser.add_argument('--compute_interactions', action='store_true')
    parser.add_argument('--compute_interaction_recovery', action='store_true')
    parser.add_argument('--protonate_ligands', action='store_true')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpus", default=8, type=int)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["pocket", "pocket_flex"], required=True)
    parser.add_argument(
        "--pocket_type", type=str, choices=["holo", "apo"], required=True
    )
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str)

    parser.add_argument("--coord_noise_std", type=float, default=0.0)

    parser.add_argument("--max_sample_iter", type=int, default=20)
    parser.add_argument("--sample_n_molecules_per_target", type=int, default=1)
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--corrector_iters", type=int, default=0)
    parser.add_argument("--use_equi_ot", action="store_true")

    parser.add_argument("--filter_valid_unique", action="store_true")

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--fixed_interactions", action="store_true")
    parser.add_argument("--interaction_inpainting", action="store_true")
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--substructure_inpainting", action="store_true")
    parser.add_argument("--substructure", type=str, default=None)
    parser.add_argument("--separate_pocket_interpolation", action="store_true")
    parser.add_argument("--separate_interaction_interpolation", action="store_true")
    parser.add_argument(
        "--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS
    )
    parser.add_argument(
        "--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL
    )
    parser.add_argument(
        "--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY
    )
    parser.add_argument(
        "--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY
    )
    parser.add_argument(
        "--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE
    )


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(args)
