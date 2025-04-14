import argparse
import datetime
import os
import warnings
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.multiprocessing
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

import flowr.scriptutil as util
from callbacks.ema import EMA, EMAModelCheckpoint
from flowr.data.data_info import GeneralInfos as DataInfos
from flowr.data.datamodules import GeometricInterpolantDM
from flowr.data.datasets import GeometricDataset
from flowr.data.interpolate import (
    ComplexInterpolant,
    GeometricInterpolant,
    GeometricNoiseSampler,
)
from flowr.data.util import Statistics
from flowr.models.fm import MolecularCFM
from flowr.models.fm_pocket import LigandPocketCFM
from flowr.models.pocket import LigandGenerator, PocketEncoder
from flowr.models.semla import EquiInvDynamics, SemlaGenerator
from flowr.util.pocket import PROLIF_INTERACTIONS

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


DEFAULT_D_MODEL = 384
DEFAULT_POCKET_D_MODEL = 256
DEFAULT_N_LAYERS = 12
DEFAULT_POCKET_N_LAYERS = 6
DEFAULT_D_MESSAGE = 64
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 128
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 96
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 64

DEFAULT_MAX_ATOMS = 183
DEFAULT_MAX_ATOMS_POCKET = 600

DEFAULT_EPOCHS = 200
DEFAULT_LR = 2e-4
DEFAULT_BATCH_COST = 512
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 10.0
DEFAULT_COORD_LOSS_WEIGHT = 1.0
DEFAULT_TYPE_LOSS_WEIGHT = 1.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_BOND_LOSS_WEIGHT = 2.0
DEFAULT_INTERACTION_LOSS_WEIGHT = 10.0
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_LR_GAMMA = 0.998
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_BUCKET_COST_SCALE = "linear"

DEFAULT_N_VALIDATION_MOLS = 64  # 64 holo only data has 64 samples, apo-holo has 51
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_POCKET_COORD_NOISE_STD_DEV = 0.0
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"

DEFAULT_CORRECTOR_ITERS = 0


# bfloat16 training produced significantly worse models than full so use default 16-bit instead
def get_precision(args):
    return "32"
    # return "16-mixed" if args.mixed_precision else "32"


def build_model(
    args, dm, dataset_info, vocab, vocab_pocket_atoms=None, vocab_pocket_res=None
):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams,
    }

    # Add 1 for the time (0 <= t <= 1 for flow matching) and potentially 1 for the atom type whether ligand or pocket
    n_atom_feats = vocab.size + 1
    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
        or args.dataset == "bindingmoad"
    ):
        n_atom_feats += 1
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    n_interaction_types = (
        len(PROLIF_INTERACTIONS) + 1
        if args.flow_interactions or args.predict_interactions
        else None
    )

    if args.arch == "semla":
        dynamics = EquiInvDynamics(
            args.d_model,
            args.d_message,
            args.n_coord_sets,
            args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_hidden=args.d_message_hidden,
            d_edge=args.d_edge,
            bond_refine=True,
            self_cond=args.self_condition,
            coord_norm=args.coord_norm,
            pocket_noise=args.pocket_noise,
            ligand_only=args.ligand_only,
        )
        egnn_gen = SemlaGenerator(
            args.d_model,
            dynamics,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            self_cond=args.self_condition,
            size_emb=args.size_emb,
            pocket_noise=args.pocket_noise,
            ligand_only=args.ligand_only,
            max_atoms=args.max_atoms,
            max_atoms_pocket=args.max_atoms_pocket,
            vocab_size_pocket_atoms=(
                vocab_pocket_atoms.size if vocab_pocket_atoms else None
            ),
            vocab_size_pocket_res=vocab_pocket_res.size if vocab_pocket_res else None,
        )
        if args.load_pretrained_ckpt:
            egnn_gen.load_state_dict(
                torch.load(args.load_pretrained_ckpt)["state_dict"], strict=False
            )
    elif args.arch == "pocket":
        fixed_equi = args.pocket_fixed_equi
        pocket_d_equi = 1 if fixed_equi else args.n_coord_sets
        pocket_d_inv = args.pocket_d_model
        pocket_n_layers = args.pocket_n_layers
        pocket_enc = PocketEncoder(
            pocket_d_equi,
            pocket_d_inv,
            args.d_message,
            pocket_n_layers,
            args.n_attn_heads,
            args.d_message_hidden,
            args.d_edge,
            vocab_pocket_atoms.size,
            n_bond_types,
            vocab_pocket_res.size,
            fixed_equi=fixed_equi,
        )
        egnn_gen = LigandGenerator(
            args.n_coord_sets,
            args.d_model,
            args.d_message,
            args.n_layers,
            args.n_attn_heads,
            args.d_message_hidden,
            args.d_edge,
            vocab.size,
            n_bond_types,
            predict_interactions=args.predict_interactions,
            flow_interactions=args.flow_interactions,
            use_lig_pocket_rbf=args.use_lig_pocket_rbf,
            use_rbf=args.use_rbf,
            use_sphcs=args.use_sphcs,
            n_interaction_types=n_interaction_types,
            n_extra_atom_feats=1,
            self_cond=args.self_condition,
            pocket_enc=pocket_enc,
            coord_skip_connect=not args.no_coord_skip_connect,
            split_continuous_discrete_time=args.split_continuous_discrete_time,
        )
    elif args.arch == "eqgat":
        from flowr.models.eqgat import EqgatGenerator

        # Hardcode for now since we only need one model size
        d_model_eqgat = 256
        n_equi_feats_eqgat = 256
        n_layers_eqgat = 12
        d_edge_eqgat = 128

        egnn_gen = EqgatGenerator(
            d_model_eqgat,
            n_layers_eqgat,
            n_equi_feats_eqgat,
            vocab.size,
            n_atom_feats,
            d_edge_eqgat,
            n_bond_types,
        )

    elif args.arch == "egnn":
        from flowr.models.egnn import VanillaEgnnGenerator

        egnn_gen = VanillaEgnnGenerator(
            args.d_model,
            args.n_layers,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
        )

    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    if args.scale_coords:
        if args.dataset == "qm9":
            coord_scale = util.QM9_COORDS_STD_DEV
        elif args.dataset == "geom-drugs":
            coord_scale = util.GEOM_COORDS_STD_DEV
        elif args.dataset == "spindr":
            coord_scale = util.PLINDER_COORDS_STD_DEV
        elif args.dataset == "crossdocked":
            coord_scale = util.CROSSDOCKED_COORDS_STD_DEV
        elif args.dataset == "kinodata":
            coord_scale = util.KINODATA_COORDS_STD_DEV
        elif args.dataset == "bindingmoad":
            coord_scale = util.BINDINGMOAD_COORDS_STD_DEV
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    else:
        coord_scale = 1.0

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "prior-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "velocity-sample":
        train_strategy = "ce"
        sampling_strategy = "velocity-sample"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    train_smiles = (
        None if args.trial_run else [mols.str_id for mols in dm.train_dataset]
    )

    print(f"Total training steps {train_steps}")

    if args.arch == "pocket":
        from flowr.models.fm_pocket import Integrator
    elif args.arch == "pocket_flex":
        from flowr.models.fm_pocket_flex import Integrator
    else:
        from flowr.models.fm import Integrator
    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        pocket_noise=args.pocket_noise,
        ligand_only=args.ligand_only,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        dataset=args.dataset,
    )

    CFM = LigandPocketCFM if args.arch == "pocket" else MolecularCFM
    fm_model = CFM(
        egnn_gen,
        vocab,
        args.lr,
        integrator,
        coord_scale=coord_scale,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        coord_loss_weight=args.coord_loss_weight,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        interaction_loss_weight=args.interaction_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        pocket_noise=args.pocket_noise,
        ligand_only=args.ligand_only,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        lr_gamma=args.lr_gamma,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_steps,
        train_smiles=train_smiles,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        remove_hs=args.remove_hs,
        save_dir=args.save_dir,
        dataset_info=dataset_info,
        data_path=args.data_path,
        flow_interactions=args.flow_interactions,
        predict_interactions=args.predict_interactions,
        interaction_inpainting=args.interaction_inpainting,
        scaffold_inpainting=args.scaffold_inpainting,
        func_group_inpainting=args.func_group_inpainting,
        linker_inpainting=args.linker_inpainting,
        fragment_inpainting=args.fragment_inpainting,
        substructure_inpainting=args.substructure_inpainting,
        mixed_uncond_inpaint=args.mixed_uncond_inpaint,
        use_t_loss_weights=args.use_t_loss_weights,
        corrector_iters=args.corrector_iters,
        **hparams,
    )
    return fm_model


def build_data_statistic(args):
    train_statistics = Statistics.get_statistics(
        os.path.join(args.data_path, "processed"),
        "train",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    val_statistics = Statistics.get_statistics(
        os.path.join(args.data_path, "processed"),
        "val",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    test_statistics = Statistics.get_statistics(
        os.path.join(args.data_path, "processed"),
        "test",
        dataset=args.dataset,
        remove_hs=args.remove_hs,
    )
    return {"train": train_statistics, "val": val_statistics, "test": test_statistics}


def build_dm(
    args,
    vocab,
    vocab_pocket_atoms=None,
    vocab_pocket_res=None,
    atom_types_distribution=None,
    bond_types_distribution=None,
):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        padded_sizes = util.QM9_BUCKET_LIMITS
    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS
    elif args.dataset == "spindr":
        coord_std = util.PLINDER_COORDS_STD_DEV
        padded_sizes = util.PLINDER_BUCKET_LIMITS
    elif args.dataset == "crossdocked":
        coord_std = util.CROSSDOCKED_COORDS_STD_DEV
        padded_sizes = util.CROSSDOCKED_BUCKET_LIMITS
    elif args.dataset == "kinodata":
        coord_std = util.KINODATA_COORDS_STD_DEV
        padded_sizes = util.KINODATA_BUCKET_LIMITS
    elif args.dataset == "bindingmoad":
        coord_std = util.BINDINGMOAD_COORDS_STD_DEV
        padded_sizes = util.BINDINGMOAD_BUCKET_LIMITS
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    if not args.scale_coords:
        coord_std = 1.0

    data_path = Path(args.data_path)

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
        or args.dataset == "bindingmoad"
    ):
        transform = partial(
            util.complex_transform,
            vocab=vocab,
            n_bonds=n_bond_types,
            coord_std=coord_std,
            remove_hs=args.remove_hs,
            pocket_noise=args.pocket_noise,
            use_interactions=args.flow_interactions
            or args.predict_interactions
            or args.interaction_inpainting,
        )
    else:
        transform = partial(
            util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std
        )

    train_dataset = GeometricDataset.load(
        data_path / "train.smol",
        dataset=args.dataset,
        transform=transform,
        remove_hs=args.remove_hs,
    )
    val_dataset = GeometricDataset.load(
        data_path / "val.smol",
        dataset=args.dataset,
        transform=transform,
        remove_hs=args.remove_hs,
    )
    if args.dataset in ["qm9", "geom-drugs"]:
        val_dataset = val_dataset.sample(args.n_validation_mols)

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"
    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"
    elif args.categorical_strategy == "prior-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "prior-sample"
    elif args.categorical_strategy == "velocity-sample":
        categorical_interpolation = "sample"
        categorical_noise = "uniform-sample"
    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported."
        )

    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if args.optimal_transport == "batch":
        batch_ot = True
    elif args.optimal_transport == "equivariant":
        equivariant_ot = True
    elif args.optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif args.optimal_transport not in ["None", "none", None]:
        raise ValueError(
            f"Unknown value for optimal_transport '{args.optimal_transport}'"
        )

    train_fixed_time = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        zero_com=True,  # args.pocket_noise in ["fix", "random"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )
    if args.dataset in ["geom-drugs", "qm9"]:
        train_interpolant = GeometricInterpolant(
            prior_sampler,
            coord_interpolation="linear",
            coord_noise_std=args.coord_noise_std_dev,
            type_interpolation=categorical_interpolation,
            bond_interpolation=categorical_interpolation,
            type_dist_temp=args.type_dist_temp,
            time_alpha=args.time_alpha,
            time_beta=args.time_beta,
            fixed_time=train_fixed_time,
            mixed_uniform_beta_time=args.mixed_uniform_beta_time,
            equivariant_ot=equivariant_ot,
            batch_ot=batch_ot,
            scale_ot=scale_ot,
        )
        eval_interpolant = GeometricInterpolant(
            prior_sampler,
            coord_interpolation="linear",
            type_interpolation=categorical_interpolation,
            bond_interpolation=categorical_interpolation,
            equivariant_ot=False,
            batch_ot=False,
            fixed_time=0.9,
        )
    else:
        train_interpolant = ComplexInterpolant(
            prior_sampler,
            ligand_coord_interpolation="linear",
            ligand_coord_noise_std=args.coord_noise_std_dev,
            ligand_type_interpolation=categorical_interpolation,
            ligand_bond_interpolation=categorical_interpolation,
            ligand_time_alpha=args.time_alpha,
            ligand_time_beta=args.time_beta,
            ligand_fixed_time=train_fixed_time,
            split_continuous_discrete_time=args.split_continuous_discrete_time,
            pocket_time_alpha=args.time_alpha,
            pocket_time_beta=args.time_beta,
            pocket_fixed_time=train_fixed_time,
            pocket_coord_noise_std=args.pocket_coord_noise_std_dev,
            rigid_pocket=args.pocket_noise == "fix",
            separate_pocket_interpolation=args.separate_pocket_interpolation,
            separate_interaction_interpolation=args.separate_interaction_interpolation,
            interaction_fixed_time=args.interaction_fixed_time,
            interaction_time_alpha=args.time_alpha,
            interaction_time_beta=args.time_beta,
            flow_interactions=args.flow_interactions,
            interaction_inpainting=args.interaction_inpainting,
            scaffold_inpainting=args.scaffold_inpainting,
            func_group_inpainting=args.func_group_inpainting,
            substructure_inpainting=args.substructure_inpainting,
            substructure=args.substructure,
            linker_inpainting=args.linker_inpainting,
            fragment_inpainting=args.fragment_inpainting,
            max_fragment_cuts=args.max_fragment_cuts,
            mixed_uncond_inpaint=args.mixed_uncond_inpaint,
            mixed_uniform_beta_time=args.mixed_uniform_beta_time,
            n_interaction_types=(
                len(PROLIF_INTERACTIONS) + 1
                if args.flow_interactions
                or args.predict_interactions
                or args.interaction_inpainting
                else None
            ),
            equivariant_ot=equivariant_ot,
            dataset=args.dataset,
            sample_mol_sizes=False,
            inference=False,
            vocab=vocab,
        )
        eval_interpolant = ComplexInterpolant(
            prior_sampler,
            ligand_coord_interpolation="linear",
            ligand_type_interpolation=categorical_interpolation,
            ligand_bond_interpolation=categorical_interpolation,
            ligand_fixed_time=0.9,
            pocket_fixed_time=0.9,
            interaction_fixed_time=0.9,
            rigid_pocket=args.pocket_noise == "fix",
            separate_pocket_interpolation=args.separate_pocket_interpolation,
            separate_interaction_interpolation=args.separate_interaction_interpolation,
            n_interaction_types=(
                len(PROLIF_INTERACTIONS) + 1
                if args.flow_interactions
                or args.predict_interactions
                or args.interaction_inpainting
                else None
            ),
            flow_interactions=args.flow_interactions,
            interaction_inpainting=args.interaction_inpainting,
            scaffold_inpainting=args.scaffold_inpainting,
            func_group_inpainting=args.func_group_inpainting,
            linker_inpainting=args.linker_inpainting,
            fragment_inpainting=args.fragment_inpainting,
            max_fragment_cuts=args.max_fragment_cuts,
            substructure_inpainting=args.substructure_inpainting,
            substructure=args.substructure,
            equivariant_ot=False,
            batch_ot=False,
            dataset=args.dataset,
            sample_mol_sizes=False,
            inference=True,
            vocab=vocab,
        )

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        val_batch_size=args.val_batch_cost,
        vocab=vocab,
        remove_hs=args.remove_hs,
        dataset=args.dataset,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        use_bucket_sampler=args.use_bucket_sampler,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )
    return dm


def build_trainer(args, model=None):
    epochs = 1 if args.trial_run else args.epochs

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    precision = get_precision(args)
    print(f"Using precision '{precision}'")

    lr_logger = LearningRateMonitor(logging_interval="step")
    mllogger = MLFlowLogger(
        experiment_name=args.dataset + "_" + args.exp_name,
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        run_id=os.environ.get("MLFLOW_RUN_ID"),
        log_model="best",
    )
    if args.wandb:
        wdblogger = WandbLogger(project=project_name, log_model="all", offline=True)
        wdblogger.watch(model, log="all")
        loggers = [wdblogger, mllogger]
    else:
        loggers = mllogger
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir)

    if args.use_ema:
        ema_callback = EMA(
            decay=args.ema_decay,
            apply_ema_every_n_steps=1,
            start_step=0,
            save_ema_weights_in_callback_state=True,
            evaluate_ema_weights_instead=True,
        )
        checkpoint_callback = EMAModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=3,
            monitor="val-fc-validity",
            mode="max",
            save_last=True,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_dir,
            save_top_k=3,
            monitor="val-fc-validity",
            mode="max",
            save_last=True,
        )
    callbacks = [
        lr_logger,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
        checkpoint_callback,
    ]
    if args.use_ema:
        callbacks.append(ema_callback)

    # Overwrite if doing a trial run
    val_check_epochs = 1 if args.trial_run else args.val_check_epochs

    from lightning.pytorch.plugins.environments import LightningEnvironment

    strategy = DDPStrategy(
        timeout=datetime.timedelta(seconds=1800 * args.gpus)
    )  # "ddp" if args.gpus > 1 else "auto"
    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus if args.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
        enable_checkpointing=True,
        accumulate_grad_batches=args.acc_batches,
        check_val_every_n_epoch=val_check_epochs,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=loggers,
        precision=precision,
        max_epochs=epochs,
        use_distributed_sampler=True,  # not args.use_bucket_sampler,
    )

    pl.seed_everything(seed=args.seed, workers=args.gpus > 1)
    return trainer


def main(args):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default setting
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
        or args.dataset == "bindingmoad"
    ):
        assert (
            args.pocket_noise is not None
        ), "pocket_noise must be set for spindr; choose from [apo, fix, random]"

    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE

    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    # vocab = util.build_vocab(remove_hs=args.remove_hs)
    vocab = util._build_vocab()
    print("Vocab complete.")

    print("Building pocket vocab...")
    if (
        args.dataset == "spindr"
        or args.dataset == "crossdocked"
        or args.dataset == "kinodata"
        or args.dataset == "kiba"
        or args.dataset == "bindingmoad"
    ):
        vocab_pocket_atoms = util._build_vocab_pocket_atoms()
        vocab_pocket_res = util._build_vocab_pocket_res()
        # vocab_pocket_atoms = util.build_vocab_pocket_atoms()
        # vocab_pocket_res = util.build_vocab_pocket_res()
    else:
        vocab_pocket_atoms = None
        vocab_pocket_res = None
    print("Vocab complete.")

    print("Loading dataset statistics...")
    statistics = build_data_statistic(args)
    dataset_info = DataInfos(statistics, vocab, args)
    atom_types_distribution = dataset_info.atom_types.float()
    bond_types_distribution = dataset_info.edge_types.float()
    print("Dataset statistics complete.")

    print("Loading datamodule...")
    dm = build_dm(
        args,
        vocab,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
        atom_types_distribution=atom_types_distribution,
        bond_types_distribution=bond_types_distribution,
    )
    print("Datamodule complete.")

    print("Building equinv model...")
    model = build_model(
        args,
        dm,
        dataset_info,
        vocab,
        vocab_pocket_atoms=vocab_pocket_atoms,
        vocab_pocket_res=vocab_pocket_res,
    )
    print("Model complete.")

    print("Fitting datamodule to model...")
    ckpt_path = None
    if args.load_ckpt is not None:
        print("Loading from checkpoint ...")

        ckpt_path = args.load_ckpt
        ckpt = torch.load(ckpt_path)
        if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != args.lr:
            print("Changing learning rate ...")
            ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = args.lr
            ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = args.lr
            ckpt_path = (
                "lr" + "_" + str(args.lr) + "_" + os.path.basename(args.load_ckpt)
            )
            ckpt_path = os.path.join(
                os.path.dirname(args.load_ckpt),
                f"retraining_with_lr{args.lr}.ckpt",
            )
            torch.save(ckpt, ckpt_path)

    trainer = build_trainer(args, model=model)
    trainer.fit(
        model,
        datamodule=dm,
        ckpt_path=ckpt_path if args.load_ckpt is not None else None,
    )
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--exp_name", type=str, default="train")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--load_ckpt", type=str, default=None)
    parser.add_argument("--load_pretrained_ckpt", type=str, default=None)
    parser.add_argument(
        "--save_dir", type=str, default="/hpfs/userws/cremej01/projects/flowr_logs"
    )
    parser.add_argument("--val_check_epochs", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")

    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--trial_run", action="store_true")

    parser.add_argument("--remove_hs", action="store_true")
    parser.add_argument("--pocket_noise", default=None, type=str)
    parser.add_argument("--separate_pocket_interpolation", action="store_true")
    parser.add_argument("--separate_interaction_interpolation", action="store_true")
    parser.add_argument("--interaction_fixed_time", type=float, default=None)
    parser.add_argument("--ligand_only", action="store_true")
    parser.add_argument("--scale_coords", action="store_true")
    parser.add_argument("--predict_interactions", action="store_true")
    parser.add_argument("--flow_interactions", action="store_true")
    parser.add_argument("--interaction_inpainting", action="store_true")
    parser.add_argument("--scaffold_inpainting", action="store_true")
    parser.add_argument("--func_group_inpainting", action="store_true")
    parser.add_argument("--fragment_inpainting", action="store_true")
    parser.add_argument("--max_fragment_cuts", type=int, default=3)
    parser.add_argument("--substructure_inpainting", action="store_true")
    parser.add_argument("--substructure", type=str, default=None)
    parser.add_argument("--linker_inpainting", action="store_true")
    parser.add_argument("--mixed_uncond_inpaint", action="store_true")
    parser.add_argument("--use_lig_pocket_rbf", action="store_true")
    parser.add_argument("--use_sphcs", action="store_true")
    parser.add_argument("--use_rbf", action="store_true")

    # Model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--pocket_d_model", type=int, default=DEFAULT_POCKET_D_MODEL)
    parser.add_argument("--pocket_fixed_equi", action="store_true")
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--pocket_n_layers", type=int, default=DEFAULT_POCKET_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument(
        "--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN
    )
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument(
        "--max_atoms_pocket", type=int, default=DEFAULT_MAX_ATOMS_POCKET
    )
    parser.add_argument("--arch", type=str, default="pocket")

    # Training args
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--val_batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--use_bucket_sampler", action="store_true")
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument(
        "--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL
    )
    parser.add_argument(
        "--coord_loss_weight", type=float, default=DEFAULT_COORD_LOSS_WEIGHT
    )
    parser.add_argument(
        "--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT
    )
    parser.add_argument(
        "--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT
    )
    parser.add_argument(
        "--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT
    )
    parser.add_argument(
        "--interaction_loss_weight",
        type=float,
        default=DEFAULT_INTERACTION_LOSS_WEIGHT,
    )
    parser.add_argument("--use_t_loss_weights", action="store_true")
    parser.add_argument(
        "--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY
    )
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--lr_gamma", type=float, default=DEFAULT_LR_GAMMA)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument(
        "--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.998)
    parser.add_argument("--self_condition", action="store_true")
    parser.add_argument("--no_coord_skip_connect", action="store_true")
    parser.add_argument("--split_continuous_discrete_time", action="store_true")
    # parser.add_argument("--mixed_precision", action="store_true")
    # parser.add_argument("--compile_model", action="store_true")
    # parser.add_argument("--distill", action="store_true")

    # Flow matching and sampling args
    parser.add_argument(
        "--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    parser.add_argument(
        "--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL
    )
    parser.add_argument(
        "--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV
    )
    parser.add_argument(
        "--pocket_coord_noise_std_dev",
        type=float,
        default=DEFAULT_POCKET_COORD_NOISE_STD_DEV,
    )
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--mixed_uniform_beta_time", action="store_true")
    parser.add_argument(
        "--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT
    )
    parser.add_argument("--corrector_iters", type=int, default=DEFAULT_CORRECTOR_ITERS)

    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
    )

    args = parser.parse_args()
    main(args)
