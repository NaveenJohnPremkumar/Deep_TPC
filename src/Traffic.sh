#!/bin/bash

model_name=GPT2WithMM2

# training one model with a context length
CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nnodes 1 --nproc-per-node 4 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.00005 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --mm_layers 5 6 7 8 9 10 11\
  --num_fusion_tokens 10 \
  --llm_ckp_dir gpt2 \
  --use_multi_gpu

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_672_96 \
  --model $model_name \
  --data custom \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.00005 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --mm_layers 5 6 7 8 9 10 11\
  --num_fusion_tokens 10 \
  --llm_ckp_dir gpt2 \
  --test_dir long_term_forecast_traffic_672_96_GPT2WithMM2_custom_sl672_ll576_tl96_lr5e-05_bt256_wd1e-05_hd1024_hl2_cosTrue_mixTrue_test_0
done

