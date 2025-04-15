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

# MAIN PATH
main_path="your_main_directory"

# CKPT PATH
ckpt_path="$main_path/your_experiment_directory"
ckpt="$ckpt_path/your_ckpt.ckpt"

# POCKET and LIG DATA
## code handles .pdb and .cif for pockets/proteins and optionally .sdf and .pdb for ligands (if any, otherwise use --ligand_file None)
## NOTE: If you provide a protein PDB/CIF file, you need to provide a ligand file as well to cut out the pocket (default: 6A cutoff - adjust below if needed).
## NOTE: If you want to run conditional generation, you need to provide a ligand file as reference. 
pdb_id="your_pdb_id"
data_path="$main_path/$pdb_id"
pdb_file="$data_path/$pdb_id.pdb"
lig_file="$data_path/your_ligand.sdf"

# SAMPLING STEPS
steps=100
sampling_steps="_$steps-steps"
#sampling_steps=""

# NOISE INJECTION
coord_noise_std=0.0
noise_inject=""
# coord_noise_std=0.2
# noise_inject="_noise-$coord_noise_std"

# N MOLECULES PER TARGET
n_molecules_per_target=1000
sample_mol_sizes="_sampled-mol-sizes"
#sample_mol_sizes="_fixed-mol-size"

# CONDITIONAL GENERATION
conditional_generation=""
#conditional_generation="_interaction-cond"
#conditional_generation="_func-group-cond"
#conditional_generation="_scaffold-cond"
#conditional_generation="_linker-cond"

# SAVE DIR
save_dir="$ckpt_path/eval_$conditional_gen$n_molecules_per_target-lig-per-target$sample_mol_sizes$noise_inject$sampling_steps$sample_strategy"

# BATCH SIZE
batch_cost=100

python -m flowr.gen.generate_from_pdb \
    --pdb_file "$pdb_file" \
    --ligand_file "$lig_file" \
    --compute_interactions \
    --compute_interaction_recovery \
    --protonate_ligands \
    --cut_pocket \
    --pocket_cutoff 6 \
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
    # --linker_inpainting \
    # --interaction_inpainting \
    # --func_group_inpainting \
    # --scaffold_inpainting \

