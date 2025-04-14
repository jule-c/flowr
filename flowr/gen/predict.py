"""
Script for generating molecules using a trained model and saving them.

Note that the script currently does not save the molecules in batches - all of the molecules are generated and then
all saved together in one Smol batch. If generating many molecules ensure you have enough memory to store them.
"""

import argparse
import pickle
import shutil
import warnings
from functools import partial
from pathlib import Path

import lightning as L
import torch

import flowr.scriptutil as util
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.datasets import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricInterpolant,
    GeometricNoiseSampler,
)
from flowr.models.fm import MolecularCFM
from flowr.models.fm_pocket import LigandPocketCFM
from flowr.models.pocket import LigandGenerator, PocketEncoder
from flowr.models.semla import EquiInvDynamics, SemlaGenerator
from flowr.train import DataInfos, build_data_statistic
from flowr.util.molrepr import GeometricMolBatch
from flowr.util.pocket import PROLIF_INTERACTIONS
from flowr.util.rdkit import write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Default script arguments
DEFAULT_SAVE_FILE = "predictions.smol"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_N_MOLECULES = 100
DEFAULT_BATCH_COST = 1
DEFAULT_BUCKET_COST_SCALE = "quadratic"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "linear"
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"


def load_model(args, vocab, vocab_pocket_atoms=None, vocab_pocket_res=None):
    checkpoint = torch.load(args.ckpt_path)
    hparams = dotdict(checkpoint["hyper_parameters"])

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    hparams["predict_interactions"] = (
        False
        if "predict_interactions" not in hparams
        else hparams["predict_interactions"]
    )  # for backwards compatibility
    hparams["interaction_inpainting"] = (
        args.interaction_inpainting
        if "interaction_inpainting" not in hparams
        else hparams["interaction_inpainting"]
    )  # for backwards compatibility

    hparams["data_path"] = args.data_path
    hparams["dataset"] = args.dataset
    hparams["save_dir"] = args.save_dir

    print("Loading dataset statistics...")
    statistics = build_data_statistic(hparams)
    dataset_info = DataInfos(statistics, vocab, hparams)
    print("Dataset statistics complete.")

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if hparams["flow_interactions"]
        or hparams["predict_interactions"]
        or hparams["interaction_inpainting"]
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
            n_extra_atom_feats=1,
            self_cond=hparams["self_cond"],
            pocket_enc=pocket_enc,
        )
    else:
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
            pocket_noise=hparams["pocket_noise"],
            ligand_only=hparams["ligand_only"],
            bond_refine=True,
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            pocket_noise=hparams["pocket_noise"],
            ligand_only=hparams["ligand_only"],
            max_atoms=hparams["max_atoms"],
            max_atoms_pocket=hparams["max_atoms_pocket"],
            vocab_size_pocket_atoms=(
                vocab_pocket_atoms.size if vocab_pocket_atoms else None
            ),
            vocab_size_pocket_res=vocab_pocket_res.size if vocab_pocket_res else None,
        )

    type_mask_index = None
    bond_mask_index = None

    if args.arch == "pocket":
        from flowr.models.fm_pocket import Integrator
    else:
        from flowr.models.fm import Integrator
    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        pocket_noise=hparams["pocket_noise"],
        ligand_only=hparams["ligand_only"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
    )

    CFM = LigandPocketCFM if args.arch == "pocket" else MolecularCFM
    fm_model = CFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        dataset_info=dataset_info,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams,
    )
    return fm_model, dataset_info


def build_dm(args, hparams, vocab, dataset_info):
    assert (
        args.dataset == hparams["dataset"]
    ), "Dataset mismatch between args and hparams"

    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    elif args.dataset == "spindr":
        coord_std = util.PLINDER_COORDS_STD_DEV
        bucket_limits = util.PLINDER_BUCKET_LIMITS
    elif args.dataset == "crossdocked":
        coord_std = util.CROSSDOCKED_COORDS_STD_DEV
        bucket_limits = util.CROSSDOCKED_BUCKET_LIMITS
    elif args.dataset == "kinodata":
        coord_std = util.KINODATA_COORDS_STD_DEV
        bucket_limits = util.KINODATA_BUCKET_LIMITS
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if hparams["coord_scale"] == 1.0:
        coord_std = 1.0

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
    ):
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
    else:
        transform = partial(
            util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std
        )

    if args.dataset_split == "train":
        if args.dataset == "crossdocked":
            dataset_path = Path(args.data_path) / "train.npz"
        else:
            dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        if args.dataset == "crossdocked":
            dataset_path = Path(args.data_path) / "val.npz"
        else:
            dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        if args.dataset == "crossdocked":
            dataset_path = Path(args.data_path) / "test.npz"
        else:
            dataset_path = Path(args.data_path) / "test.smol"

    dataset = GeometricDataset.load(
        dataset_path,
        dataset=hparams["dataset"],
        transform=transform,
        remove_hs=hparams["remove_hs"],
    )

    if args.dataset in ["qm9", "geom-drugs"]:
        if args.dataset_split == "test":
            n_molecules = len(dataset)
        else:
            n_molecules = args.n_molecules
        dataset = dataset.sample(n_molecules, replacement=True)

    if args.sample_n_molecules_per_target > 1:
        dataset = dataset.sample_n_molecules_per_target(
            args.sample_n_molecules_per_target
        )

    type_mask_index = None
    bond_mask_index = None

    if args.dataset in ["qm9", "geom-drugs"]:
        prior_sampler = GeometricNoiseSampler(
            vocab.size,
            n_bond_types,
            coord_noise="gaussian",
            type_noise=hparams["val-prior-type-noise"],
            bond_noise=hparams["val-prior-bond-noise"],
            zero_com=True,
            type_mask_index=type_mask_index,
            bond_mask_index=bond_mask_index,
            atom_types_distribution=dataset_info.atom_types.float(),
            bond_types_distribution=dataset_info.edge_types.float(),
        )
        eval_interpolant = GeometricInterpolant(
            prior_sampler,
            coord_interpolation="linear",
            type_interpolation=hparams["val-type-interpolation"],
            bond_interpolation=hparams["val-bond-interpolation"],
            equivariant_ot=False,
            batch_ot=False,
        )
    elif args.dataset in ["spindr", "crossdocked", "kinodata", "kiba"]:
        prior_sampler = GeometricNoiseSampler(
            vocab.size,
            n_bond_types,
            coord_noise="gaussian",
            type_noise=hparams["val-ligand-prior-type-noise"],
            bond_noise=hparams["val-ligand-prior-bond-noise"],
            zero_com=True,
            type_mask_index=type_mask_index,
            bond_mask_index=bond_mask_index,
            atom_types_distribution=dataset_info.atom_types.float(),
            bond_types_distribution=dataset_info.edge_types.float(),
        )
        if args.pocket_time is not None:
            assert (
                args.separate_pocket_interpolation
            ), "Setting a pocket time requires a separate pocket interpolation"
        if args.interaction_time is not None:
            assert (
                args.separate_interaction_interpolation
            ), "Setting an interaction time requires a separate interaction interpolation"
        eval_interpolant = ComplexInterpolant(
            prior_sampler,
            ligand_coord_interpolation="linear",
            ligand_type_interpolation="unmask",
            ligand_bond_interpolation="unmask",
            ligand_fixed_time=args.ligand_time,
            pocket_fixed_time=args.pocket_time,
            interaction_fixed_time=args.interaction_time,
            rigid_pocket=hparams["pocket_noise"] == "fix",
            separate_pocket_interpolation=args.separate_pocket_interpolation,
            separate_interaction_interpolation=args.separate_interaction_interpolation,
            n_interaction_types=(
                len(PROLIF_INTERACTIONS) + 1
                if hparams["flow_interactions"]
                or hparams["predict_interactions"]
                or hparams["interaction_inpainting"]
                else None
            ),
            flow_interactions=hparams["flow_interactions"],
            interaction_inpainting=hparams["interaction_inpainting"],
            dataset=args.dataset,
            sample_mol_sizes=args.sample_mol_sizes,
            equivariant_ot=False,
            batch_ot=False,
            inference=True,
        )
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        val_batch_size=args.batch_cost,
        dataset=args.dataset,
        vocab=vocab,
        remove_hs=hparams["remove_hs"],
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )
    return dm, transform


def dm_from_ckpt(args, vocab, dataset_info):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    hparams["flow_interactions"] = (
        hparams["flow_interactions"] if "flow_interactions" in hparams else False
    )  # for backwards compatibility
    hparams["predict_interactions"] = (
        hparams["predict_interactions"] if "predict_interactions" in hparams else False
    )  # for backwards compatibility
    hparams["interaction_inpainting"] = (
        hparams["interaction_inpainting"]
        if "interaction_inpainting" in hparams
        else args.interaction_inpainting
    )  #
    dm, transform = build_dm(args, hparams, vocab, dataset_info)
    return dm, transform


def generate_smol_mols(output, model):
    coords = output["coords"]
    atom_dists = output["atomics"]
    bond_dists = output["bonds"]
    charge_dists = output["charges"]
    masks = output["mask"]

    mols = model.builder.smol_from_tensors(
        coords, atom_dists, masks, bond_dists=bond_dists, charge_dists=charge_dists
    )
    return mols


def save_predictions(args, raw_outputs, model):
    # Generate GeometricMols and then combine into one GeometricMolBatch
    mol_lists = [generate_smol_mols(output, model) for output in raw_outputs]
    mols = [mol for mol_list in mol_lists for mol in mol_list]
    batch = GeometricMolBatch.from_list(mols)

    save_path = Path(args.save_dir) / args.save_file
    batch_bytes = batch.to_bytes()
    save_path.write_bytes(batch_bytes)


def save_data(args, outputs, validities):
    save_path = Path(args.save_dir)
    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=False)

    valid_gen_ligs = outputs["valid_gen_ligs"]
    valid_gen_pdbs = outputs["gen_pdbs"]

    for lig, pdb in zip(valid_gen_ligs, valid_gen_pdbs):
        target = pdb.stem
        write_sdf_file(
            lig,
            save_path / f"gen_lig_{target}.sdf",
        )
        shutil.move(pdb, save_path / f"gen_pdb_{target}.pdb")

    with open(save_path / "validities.pickle", "wb") as f:
        pickle.dump(validities, f)


def main(args):
    print(f"Running prediction script for {args.n_molecules} molecules...")
    print(f"Using model stored at {args.ckpt_path}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
    ):
        vocab_pocket_atoms = util.build_vocab_pocket_atoms(
            pocket_noise=args.pocket_noise
        )
        vocab_pocket_res = util.build_vocab_pocket_res(pocket_noise=args.pocket_noise)
    print("Vocab complete.")

    print("Loading model...")
    model, dataset_info = load_model(
        args,
        vocab,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
    )
    print("Model complete.")

    print("Loading datamodule...")
    dm, transform = dm_from_ckpt(args, vocab, dataset_info)
    print("Datamodule complete.")

    print("Initialising metrics...")
    metrics, metrics_sbdd, _ = util.init_metrics(
        args, model, transform, dataset_info=dataset_info
    )
    print("Metrics complete.")

    print("Running generation...")
    if (
        args.dataset == "crossdocked"
        or args.dataset == "spindr"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
    ):
        outputs, validities = util.generate_complexes(
            model,
            dm,
            args.integration_steps,
            args.ode_sampling_strategy,
            arch=args.arch,
            vocab=vocab,
            dataset_info=dataset_info,
        )
        molecules = outputs["valid_gen_ligs"]
    else:
        molecules, raw_outputs = util.generate_molecules(
            model, dm, args.integration_steps, args.ode_sampling_strategy
        )
    print("Generation complete.")

    if (
        args.dataset == "crossdocked"
        or args.dataset == "spindr"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
    ):
        print(f"Saving data to {args.save_dir}")
        save_data(args, outputs, validities)
    else:
        print(f"Saving predictions to {args.save_dir}/{args.save_file}")
        save_predictions(args, raw_outputs, model)
    print("Complete.")

    print("Calculating generative metrics...")
    results = util.calc_metrics_(molecules, metrics)
    util.print_results(results)
    print("Generation script complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", type=str, choices=["pocket", "semla"], required=True)
    parser.add_argument(
        "--pocket_noise", type=str, choices=["fix", "random", "apo"], required=True
    )
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)

    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--sample_n_molecules_per_target", type=int, default=1)
    parser.add_argument("--sample_mol_sizes", action="store_true")
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--fixed_interactions", action="store_true")
    parser.add_argument("--interaction_inpainting", action="store_true")
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
    main(args)
