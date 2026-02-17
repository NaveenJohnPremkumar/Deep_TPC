#!/usr/bin/env bash
set -e                                  # stop on the first error
export CUDA_VISIBLE_DEVICES=6           # choose a single GPU; adjust as needed

model_name=AutoTimes_LlamaMM
ckpt_dir_base=/scratch3/home/fbellos/research/AutoTimesWithMM/checkpoints
llm_ckpt=/scratch3/home/fbellos/llama-2-7b-hf      # local Llama-2-7B path

# --------------------------------------------------------------------
# mm-layer configurations
# first 1-3 layers (0,0-1,0-1-2)  +  last 1-3 layers (31,30-31,29-30-31)
# If your base model has a different depth, edit these indices.
# --------------------------------------------------------------------
mm_cfgs=(
  "0"
  "0 1"
  "0 1 2"
  "31"
  "30 31"
  "29 30 31"
)

# Learning-rate sweep (edit or extend as you like)
lrs=(1e-4 5e-4 1e-3)

for mm_layers in "${mm_cfgs[@]}"; do
  for lr in "${lrs[@]}"; do

    # Replace spaces with dashes so the value is filename-friendly
    mm_tag=$(echo "${mm_layers}" | tr ' ' '-')
    run_stamp=$(date +%Y%m%d_%H%M%S)
    run_name="ETTh1_${model_name}_mm${mm_tag}_lr${lr}_${run_stamp}"
    ckpt_dir="${ckpt_dir_base}/${run_name}"

    echo -e "\n=== Running ${run_name} ==="
    mkdir -p "${ckpt_dir}"

    python -u run.py \
      --task_name          long_term_forecast \
      --is_training        1 \
      --root_path          ./dataset/ETT-small/ \
      --data_path          ETTh1.csv \
      --model_id           "${run_name}" \
      --model              "${model_name}" \
      --data               ETTh1 \
      --seq_len            672 \
      --label_len          576 \
      --token_len          96  \
      --test_seq_len       672 \
      --test_label_len     576 \
      --test_pred_len      96  \
      --batch_size         256 \
      --learning_rate      "${lr}" \
      --mlp_hidden_layers  2 \
      --train_epochs       10 \
      --use_amp \
      --cosine \
      --tmax              10 \
      --mix_embeds \
      --drop_last \
      --mm_layers          ${mm_layers} \
      --num_fusion_tokens  10 \
      --patience           3 \
      --hidden_size        4096 \
      --llm_ckp            "${llm_ckpt}" \
      --gpu               0 \
      --output_dir        "${ckpt_dir}"

    # ----------------------------------------------------------------
    # Uncomment the following line to pause a few seconds between runs
    # sleep 5
    # ----------------------------------------------------------------

  done
done