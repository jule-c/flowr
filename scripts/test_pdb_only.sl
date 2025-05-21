#!/bin/bash
#SBATCH -J SamplePDB
#SBATCH --time=00-00:29:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/hpfs/userws/cremej01/slurm_outs/generate-sbdd_%j.out
#SBATCH --error=/hpfs/userws/cremej01/slurm_outs/generate-sbdd_%j.err

cd /hpfs/userws/cremej01/projects/flowr_test
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/mamba.sh
source /hpfs/userws/cremej01/projects/mambaforge/etc/profile.d/conda.sh
conda activate flowr

export PYTHONPATH="/hpfs/userws/cremej01/projects/flowr_test"

# COMPUTE
num_gpus=1
num_workers=12

# MAIN PATH
main_path="/hpfs/userws/cremej01/projects"

# CKPT PATH
ckpt_path="$main_path/plinder/flowr_logs/pocket-encoder_128-equi_8-gpus_bs-3800_no-deduplication_bucket-sampler_full-inpainting_rbf_lr-5_OLD"
ckpt="$ckpt_path/epoch=424-step=124950-EMA.ckpt"


# PDB DATA
data="5yea"
data_path="$main_path/$data"

# SAMPLING STEPS
steps=100
sampling_steps="_$steps-steps"
#sampling_steps=""

# NOISE INJECTION
# coord_noise_std=0.0
# noise_inject=""
coord_noise_std=0.2
noise_inject="_noise-$coord_noise_std"

# N MOLECULES PER TARGET
n_molecules_per_target=20
#sample_mol_sizes="_sampled-mol-sizes"
sample_mol_sizes="_fixed-mol-size"

# CONDITIONAL GENERATION
conditional_generation=""
#conditional_generation="_interaction-cond"
#conditional_generation="_func-group-cond"
#conditional_generation="_scaffold-cond"
#conditional_generation="_linker-cond"

# SAVE DIR
save_dir="$data_path/test_pdb_only"

# BATCH SIZE
batch_cost=50

python -m flowr.gen.generate_from_pdb \
    --pdb_file "/hpfs/userws/cremej01/projects/5yea/processed_scaffold-cond_fixed-mol-size_100-steps/ref_pdbs/5yea__2__1.B__1.G_with_hs.pdb" \
    --protonate_generated_ligands \
    --gpus "$num_gpus" \
    --batch_cost $batch_cost \
    --arch pocket \
    --pocket_type holo \
    --ckpt_path "$ckpt" \
    --save_dir "$save_dir" \
    --max_sample_iter 20 \
    --coord_noise_std $coord_noise_std \
    --sample_n_molecules_per_target $n_molecules_per_target \
    --categorical_strategy uniform-sample \
    --filter_valid_unique \
    --sample_mol_sizes \
    --num_heavy_atoms 20 \
    # --linker_inpainting \
    # --interaction_inpainting \
    # --func_group_inpainting \
    # --scaffold_inpainting \

