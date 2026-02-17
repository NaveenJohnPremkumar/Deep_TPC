#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
model_name=AutoTimes_LlamaMM

# # export CUDA_LAUNCH_BLOCKING=1

# CKPT_ROOT=/scratch3/home/fbellos/research/AutoTimesWithMM/checkpoints
# STAMP=$(date +%Y%m%d_%H%M%S)
# RUN_NAME="ETTh1_${model_name}_${STAMP}"
# TEST_DIR="${CKPT_ROOT}/${RUN_NAME}"   # you control this name

# (Optional) ensure dir exists if you plan to copy results into it
# mkdir -p "$TEST_DIR"
# training one model with a context length
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 48 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --mm_layers 31 \
  --num_fusion_tokens 10 \
  --patience 3 \
  --hidden_size 4096 \
  --llm_ckp /scratch3/home/fbellos/llama-2-7b-hf
  # --mm_layers 0 2 4 6 8 10 \

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_672_96 \
  --model $model_name \
  --data ETTh1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 48 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --mm_layers 31 \
  --num_fusion_tokens 10 \
  --patience 3 \
  --hidden_size 4096 \
  --llm_ckp /scratch3/home/fbellos/llama-2-7b-hf \
  --test_dir /scratch3/home/fbellos/research/AutoTimesWithMM/checkpoints/long_term_forecast_ETTh1_672_96_AutoTimes_LlamaMM_ETTh1_sl672_ll576_tl96_lr0.0001_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0
done

# model_name=AutoTimes_Llama

# # training one model with a context length
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_672_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --seq_len 672 \
#   --label_len 576 \
#   --token_len 96 \
#   --test_seq_len 672 \
#   --test_label_len 576 \
#   --test_pred_len 96 \
#   --batch_size 256 \
#   --learning_rate 0.0005 \
#   --mlp_hidden_layers 0 \
#   --train_epochs 10 \
#   --use_amp \
#   --gpu 0 \
#   --cosine \
#   --tmax 10 \
#   --mix_embeds \
#   --llm_ckp /scratch3/home/fbellos/llama-2-7b-hf \
#   --drop_last

# # testing the model on all forecast lengths
# for test_pred_len in 96 192 336 720
# do
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 0 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_672_96 \
#   --model $model_name \
#   --data ETTh1 \
#   --seq_len 672 \
#   --label_len 576 \
#   --token_len 96 \
#   --test_seq_len 672 \
#   --test_label_len 576 \
#   --test_pred_len $test_pred_len \
#   --batch_size 256 \
#   --learning_rate 0.0005 \
#   --mlp_hidden_layers 0 \
#   --train_epochs 10 \
#   --use_amp \
#   --gpu 0 \
#   --cosine \
#   --tmax 10 \
#   --mix_embeds \
#   --drop_last \
#   --llm_ckp /scratch3/home/fbellos/llama-2-7b-hf \
#   --test_dir long_term_forecast_ETTh1_672_96_AutoTimes_Llama_ETTh1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0
# done