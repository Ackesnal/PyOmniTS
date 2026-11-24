#!/usr/bin/env bash
# Derived from this model's MIMIC_IV.sh; 36 h input + next 3 observed rows.
# Optional overrides: MELODY_PYTHON, MELODY_DATASET_ROOT; extra CLI args follow defaults.
set -eu
melody_project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$melody_project_dir"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/home/s4695741/CHORD_MELODY/tmp
export XDG_CACHE_HOME=/home/s4695741/CHORD_MELODY/cache/runtime
export TORCH_HOME=/home/s4695741/CHORD_MELODY/cache/torch
export HF_HOME=/home/s4695741/CHORD_MELODY/cache/huggingface
export MPLCONFIGDIR=/home/s4695741/CHORD_MELODY/cache/matplotlib
use_multi_gpu=0
if [ $use_multi_gpu -eq 0 ]; then
    launch_command="${MELODY_PYTHON:-python}"
else
    launch_command="accelerate launch"
fi

. "$(dirname "$(readlink -f "$0")")/../globals.sh" # Import shared information from scripts/globals.sh

dataset_name=$(basename "$0" .sh) # file name
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name" # Get dataset information from scripts/globals.sh

model_name="$(basename "$(dirname "$(readlink -f "$0")")")" # folder name
model_id=$model_name

seq_len=2160
for pred_len in 3; do
    $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --use_multi_gpu $use_multi_gpu \
    --d_model 256 \
    --patch_len 4 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --dataset_root_path "$dataset_root_path" \
    --model_id $model_id \
    --model_name $model_name \
    --dataset_name $dataset_name \
    --dataset_id $dataset_id \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in $n_variables \
    --dec_in $n_variables \
    --c_out $n_variables \
    --train_epochs 300 \
    --patience 10 \
    --val_interval 1 \
    --itr 5 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --skip_test 0 \
    --random_seed 20261006 \
    --wandb 0 \
    --sweep 0 \
    "$@"
done

