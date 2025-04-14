#!/bin/bash

num_workers=100

dataset="spindr"
main_path="your_main_directory"
data_path="$main_path/$dataset"

ckpt_path="$main_path/your_experiment_directory"

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

# SAVE DIR
save_dir="$ckpt_path/eval_$conditional_gen$n_molecules_per_target-lig-per-target$sample_mol_sizes$noise_inject$sampling_steps$sample_strategy"

# RUN evaluation
python -m flowr.eval.evaluate_metrics \
    --num_workers "$num_workers" \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --save_dir "$save_dir" \
    --multiple_files \
    --remove_hs \
    # --valid_unique \

python -m flowr.eval.evaluate_interactions \
    --num_workers "$num_workers" \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --save_dir "$save_dir" \
    --return_interaction_list \
    --multiple_files \
    --remove_hs \
    # --valid_unique \
