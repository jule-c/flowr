#!/bin/bash
#SBATCH -J SampleArray
#SBATCH --time=00-01:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=12
#SBATCH --partition=your_partition
#SBATCH --gres=gpu:1
#SBATCH --array=1-2
#SBATCH --output=your_output_path/generate-sbdd_%j.out
#SBATCH --error=your_output_path/generate-sbdd_%j.err

cd your_flowr_directory
source your_mamba_path/mamba.sh
source your_conda_path/conda.sh
conda activate flowr

export PYTHONPATH="your_flowr_directory"

num_gpus=2
num_workers=12

dataset="spindr"
main_path="your_main_directory"
data_path="$main_path/$dataset"

ckpt_path="$main_path/your_experiment_directory"
ckpt="$ckpt_path/your_ckpt.ckpt"

# SAMPLING STEPS
steps=100
sampling_steps="_$steps-steps"

# SAMPLING STRATEGY
sampling_strategy="linear"
#sampling_strategy="log"
sample_strategy="_$sampling_strategy-sampling-strategy"

# NOISE INJECTION
coord_noise_std=0.0
noise_inject=""
# coord_noise_std=0.2
# noise_inject="_noise-$coord_noise_std"

# N MOLECULES PER TARGET
n_molecules_per_target=100
sample_mol_sizes="_sampled-mol-sizes"
#sample_mol_sizes="_fixed-mol-size"

# CONDITIONAL GENERATION
conditional_gen=""
#conditional_gen="interaction-cond_"
#conditional_gen="func-group-cond_"
#conditional_gen="scaffold-cond_"
#conditional_gen="linker-cond_"

# BATCH SIZE
batch_cost=100 # batch size for each GPU (H100 with 80GB can easily handle 100)

# SAVE DIR
save_dir="$ckpt_path/eval_$conditional_gen$n_molecules_per_target-lig-per-target$sample_mol_sizes$noise_inject$sampling_steps$sample_strategy"

# RUN generation
python -m flowr.gen.generate_from_smol \
    --mp_index "${SLURM_ARRAY_TASK_ID}" \
    --gpus "$num_gpus" \
    --batch_cost $batch_cost \
    --arch pocket \
    --pocket_noise fix \
    --dataset_split test \
    --ckpt_path "$ckpt" \
    --data_path "$data_path" \
    --dataset "$dataset" \
    --save_dir "$save_dir" \
    --max_sample_iter 20 \
    --coord_noise_std $coord_noise_std \
    --sample_n_molecules_per_target $n_molecules_per_target \
    --integration_steps $steps \
    --ode_sampling_strategy $sampling_strategy \
    --sample_mol_sizes \
    # --use_equi_ot \
    # --linker_inpainting \
    # --scaffold_inpainting \
    # --func_group_inpainting \
    # --interaction_inpainting \
    # --filter-valid-unique \

