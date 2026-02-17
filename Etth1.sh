#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
model_name=GPT2WithMM

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
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 2 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --mm_layers 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 \
  --num_fusion_tokens 10 \
  --patience 3 \
  --llm_ckp_dir /scratch3/home/fbellos/gpt2-large \
  --hidden_size 1280 

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
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 2 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --mm_layers 30 31 32 33 34 35 \
  --num_fusion_tokens 10 \
  --llm_ckp_dir /scratch3/home/fbellos/gpt2-large \
  --patience 3 \
  --hidden_size 1280 \
  --test_dir /scratch3/home/fbellos/research/AutoTimesWithMM/checkpoints/long_term_forecast_ETTh1_672_96_GPT2WithMM_ETTh1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl2_cosTrue_mixTrue_test_0
done
#   # --mm_layers 5 6 7 8 9 10 11 \

# # for test_pred_len in 192 336
# # do
# # python -u run.py \
# #   --task_name long_term_forecast \
# #   --is_training 0 \
# #   --root_path ./dataset/ETT-small/ \
# #   --data_path ETTh1.csv \
# #   --model_id ETTh1_672_96 \
# #   --model $model_name \
# #   --data ETTh1 \
# #   --seq_len 672 \
# #   --label_len 576 \
# #   --token_len 192 \
# #   --test_seq_len 672 \
# #   --test_label_len 576 \
# #   --test_pred_len $test_pred_len \
# #   --batch_size 256 \
# #   --learning_rate 0.0005 \
# #   --mlp_hidden_layers 0 \
# #   --train_epochs 20 \
# #   --use_amp \
# #   --gpu 0 \
# #   --cosine \
# #   --tmax 10 \
# #   --mix_embeds \
# #   --drop_last \
# #   --mm_layers 5 6 7 8 9 10 11 \
# #   --num_fusion_tokens 20 \
# #   --llm_ckp_dir gpt2 \
# #   --test_dir long_term_forecast_ETTh1_672_96_GPT2WithMM_ETTh1_sl672_ll576_tl96_lr0.001_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0
# # done


# # ###########original autotimes#####
# model_name=AutoTimes_Gpt2

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
#   --batch_size 128 \
#   --learning_rate 0.0005 \
#   --mlp_hidden_layers 0 \
#   --train_epochs 10 \
#   --use_amp \
#   --gpu 0 \
#   --cosine \
#   --tmax 10 \
#   --llm_ckp_dir /scratch3/home/fbellos/gpt2-large \
#   --hidden_size 1280 \
#   --drop_last

# # # testing the model on all forecast lengths
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
#   --batch_size 128 \
#   --learning_rate 0.0005 \
#   --mlp_hidden_layers 0 \
#   --train_epochs 10 \
#   --use_amp \
#   --gpu 0 \
#   --cosine \
#   --tmax 10 \
#   --mix_embeds \
#   --drop_last \
#   --llm_ckp_dir /scratch3/home/fbellos/gpt2-large \
#   --hidden_size 1280 \
#   --test_dir long_term_forecast_ETTh1_672_96_AutoTimes_Gpt2_ETTh1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl2_cosTrue_mixTrue_test_0
# done
