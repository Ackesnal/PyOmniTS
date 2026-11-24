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

. "$(dirname "$(readlink -f "$0")")/../globals.sh"

dataset_name=$(basename "$0" .sh)
dataset_subset_name=""
dataset_id=$dataset_name
get_dataset_info "$dataset_name" "$dataset_subset_name"

model_name="$(basename "$(dirname "$(readlink -f "$0")")")"
model_id=$model_name

seq_len=2160
d_model=192
for pred_len in 3; do
    CUDA_VISIBLE_DEVICES=1 $launch_command main.py \
    --is_training 1 \
    --collate_fn "collate_fn" \
    --loss "MSE" \
    --dropout 0.1 \
    --d_model $d_model \
    --d_ff 768 \
    --n_layers 2 \
    --n_heads 6 \
    --gate_warmup_epochs 5 \
    --use_multi_gpu $use_multi_gpu \
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
    --train_epochs 100 \
    --patience 10 \
    --val_interval 1 \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 1e-3
done
