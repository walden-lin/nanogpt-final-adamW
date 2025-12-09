#!/usr/bin/env bash
set -euo pipefail

# Grid runner for optimizer comparisons with GQA enabled.
# Adjust the hyperparameters below to change model size or training length.

ROOT_OUT="out"
MAX_ITERS=2000
BLOCK_SIZE=128
BATCH_SIZE=32
N_LAYER=6
N_HEAD=6
N_HEADGROUP=2   # enable grouped-query attention (GQA)
N_EMBD=384
DEVICE=${DEVICE:-cuda}

LR_VALUES=("1e-3" "5e-4")
WEIGHT_DECAYS=("0.01" "0.1")
L2_VALUES=("0.01" "0.1")
SCHEDULES=("constant" "cosine")
WARM_RESTARTS=("false" "true")

run() {
  echo "Launching: $*"
  "$@"
}

# AdamW runs
for lr in "${LR_VALUES[@]}"; do
  for wd in "${WEIGHT_DECAYS[@]}"; do
    for sched in "${SCHEDULES[@]}"; do
      for wr in "${WARM_RESTARTS[@]}"; do
        if [[ "$sched" != "cosine" && "$wr" == "true" ]]; then
          continue
        fi
        run_name="adamw_${sched}"
        [[ "$wr" == "true" ]] && run_name="${run_name}_wr"
        run_name="${run_name}_lr${lr}_wd${wd}"
        run python train.py \
          --out_dir="${ROOT_OUT}" \
          --run_name="${run_name}" \
          --optim_mode=adamw \
          --lr_schedule="${sched}" \
          --warm_restarts="${wr}" \
          --learning_rate="${lr}" \
          --weight_decay="${wd}" \
          --block_size=${BLOCK_SIZE} \
          --batch_size=${BATCH_SIZE} \
          --n_layer=${N_LAYER} \
          --n_head=${N_HEAD} \
          --n_headgroup=${N_HEADGROUP} \
          --n_embd=${N_EMBD} \
          --dropout=0.0 \
          --max_iters=${MAX_ITERS} \
          --eval_interval=200 \
          --log_interval=10 \
          --device="${DEVICE}" \
          --attention_variant=gqa
      done
    done
  done
done

# Adam + L2 runs (no warm restarts)
for lr in "${LR_VALUES[@]}"; do
  for l2 in "${L2_VALUES[@]}"; do
    for sched in "${SCHEDULES[@]}"; do
      run_name="adaml2_${sched}_lr${lr}_l2${l2}"
      run python train.py \
        --out_dir="${ROOT_OUT}" \
        --run_name="${run_name}" \
        --optim_mode=adam_l2 \
        --lr_schedule="${sched}" \
        --warm_restarts=false \
        --learning_rate="${lr}" \
        --l2_lambda="${l2}" \
        --weight_decay=0.0 \
        --block_size=${BLOCK_SIZE} \
        --batch_size=${BATCH_SIZE} \
        --n_layer=${N_LAYER} \
        --n_head=${N_HEAD} \
        --n_headgroup=${N_HEADGROUP} \
        --n_embd=${N_EMBD} \
        --dropout=0.0 \
        --max_iters=${MAX_ITERS} \
        --eval_interval=200 \
        --log_interval=10 \
        --device="${DEVICE}" \
        --attention_variant=gqa
    done
  done
done

echo "All experiment launches completed."
