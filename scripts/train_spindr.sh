#!/bin/sh
# Modify this scripts to your needs and run it to train a model on the Spindr dataset.

main_path="your_main_directory"
exp_name="your_experiment_name" # This is also the directory name for the logs

dataset="spindr"
data_path="$main_path/$dataset"
save_dir="$main_path/$exp_name"

python -m flowr.train \
    --gpus 8 \
    --use_bucket_sampler \
    --bucket_cost_scale linear \
    --batch_cost 3000 \
    --val_batch_cost 10 \
    --d_model 512 \
    --d_edge 256 \
    --n_coord_sets 256 \
    --pocket_d_model 384 \
    --pocket_n_layers 6 \
    --arch pocket \
    --pocket_noise fix \
    --epochs 400 \
    --val_check_epochs 1 \
    --self_condition \
    --max_atoms_pocket 600 \
    --dataset "$dataset" \
    --exp_name "$exp_name" \
    --data_path "$data_path" \
    --save_dir "$save_dir" \
    --use_ema \
    --ema_decay 0.999 \
    --lr 5.0e-4 \
    --lr_schedule exponential \
    --lr_gamma 0.998 \
    --use_lig_pocket_rbf \
    --remove_hs \
    --optimal_transport equivariant \
    --categorical_strategy uniform-sample \
    --mixed_uncond_inpaint \
    --interaction_inpainting \
    --scaffold_inpainting \
    --func_group_inpainting \
    # --linker_inpainting \