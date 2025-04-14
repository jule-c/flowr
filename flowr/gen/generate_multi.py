import argparse
import time
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path

import lightning as L
import numpy as np
import torch
from tqdm import tqdm

import flowr.scriptutil as util
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.datasets import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricNoiseSampler,
)
from flowr.predict_multi import load_model
from flowr.scriptutil import generate_n_ligands
from flowr.util.pocket import PROLIF_INTERACTIONS

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


def load_dm(args, hparams, vocab, dataset_info):
    bucket_limits = util.PLINDER_BUCKET_LIMITS

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
    dataset_path = Path(args.data_path) / "test.smol"
    dataset = GeometricDataset.load(
        dataset_path,
        dataset=hparams["dataset"],
        transform=transform,
        remove_hs=hparams["remove_hs"],
    )
    dataset = dataset.split(
        idx=args.mp_index, n_chunks=args.gpus
    ).sample_n_molecules_per_target(args.sample_n_molecules_per_target)

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
    if args.use_equi_ot:
        assert (
            args.interaction_inpainting
        ), "Equivariant OT while sampling only allowed when using inpainting on reference sized molecules"

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
        dataset=args.dataset,
        sample_mol_sizes=args.sample_mol_sizes,
        equivariant_ot=args.use_equi_ot,
        vocab=vocab,
        inference=True,
    )

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
    return dm


def create_list_defaultdict():
    return defaultdict(list)


def evaluate(args):
    # load hyperparameter
    print(f"Using model stored at {args.ckpt_path}")

    L.seed_everything(args.seed)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Loading model...")
    model, hparams, dataset_info, vocab, vocab_pocket_atoms, vocab_pocket_res = (
        load_model(
            args,
        )
    )
    model = model.to("cuda")
    model.eval()
    print("Model complete.")

    # Load the data
    dm = load_dm(args, hparams, vocab, dataset_info)
    assert (
        args.batch_cost == args.sample_n_molecules_per_target
    ), "Iterating over every target step by step sampling N ligands each"
    test_dl = dm.test_dataloader()

    out_dict = defaultdict(list)

    print("\nStarting sampling...\n")
    for i, batch in tqdm(enumerate(test_dl), desc="Sampling ligands"):

        start = time.time()
        out = generate_n_ligands(
            args, hparams, model, batch, batch_idx=f"{i}_{args.mp_index}"
        )
        run_time = time.time() - start
        print(
            f"\n Run time={round(run_time, 2)} for {len(out['gen_ligs'])} molecules \n"
        )

        torch.cuda.empty_cache()

        # Save the generated ligands
        out_dict["gen_ligs"].append(out["gen_ligs"])
        out_dict["ref_ligs"].append(out["ref_ligs"])
        out_dict["ref_ligs_with_hs"].append(out["ref_ligs_with_hs"])
        out_dict["ref_pdbs"].append(out["ref_pdbs"])
        out_dict["ref_pdbs_with_hs"].append(out["ref_pdbs_with_hs"])
        # Time for the sampling process
        out_dict["time_per_pocket"].append(run_time)

    # Save out_dict as pickle file
    if args.filter_valid_unique:
        predictions = (
            Path(args.save_dir) / f"predictions_multi_valid_unique_{args.mp_index}.pt"
        )
    else:
        predictions = Path(args.save_dir) / f"predictions_multi_{args.mp_index}.pt"
    torch.save(out_dict, str(predictions))
    print(f"Samples saved as {str(predictions)}")

    print(
        f"Time per pocket: {np.mean(out_dict['time_per_pocket']):.3f} \pm "
        f"{np.std(out_dict['time_per_pocket']):.2f}"
    )
    print("Sampling finished.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mp_index', default=0, type=int)
    parser.add_argument("--gpus", default=8, type=int)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--arch", type=str, choices=["pocket", "semla"], required=True)
    parser.add_argument(
        "--pocket_noise", type=str, choices=["fix", "random", "apo"], required=True
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

    parser.add_argument("--filter_valid_unique", action="store_true")

    parser.add_argument("--batch_cost", type=int)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--ligand_time", type=float, default=None)
    parser.add_argument("--pocket_time", type=float, default=None)
    parser.add_argument("--interaction_time", type=float, default=None)
    parser.add_argument("--resampling_steps", type=int, default=None)
    parser.add_argument("--interaction_inpainting", action="store_true")
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--use_equi_ot", action="store_true")
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
