#!/bin/bash
#SBATCH --job-name=GPt2mmtraffic
#SBATCH --partition=spgpu
#SBATCH --account=jjcorso_owned1
#SBATCH --time=00-10:00:00
#SBATCH --gpus=8
#SBATCH --gres=gpu:8
#SBATCH --nodes=1     # Running on one node
#SBATCH --ntasks=8    # Running four tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --output=/home/naveenjp/outputs/gpt2mmtraffic.log
#SBATCH --error=/home/naveenjp/errors/gpt2mmtraffic.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=naveenjp@umich.edu

model_name=GPT2WithMM

module load python/3.10.4
module load numpy

# training one model with a context length
torchrun --nnodes 1 --nproc-per-node 8 run.py \
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
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --use_multi_gpu \
  --mm_layers 0 2 4 6 8 10 \
  --num_fusion_tokens 20 \
  --llm_ckp_dir gpt2

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
  --learning_rate 0.0001 \
  --weight_decay 0.00001 \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --mm_layers 0 2 4 6 8 10 \
  --num_fusion_tokens 20 \
  --llm_ckp_dir gpt2
  --test_dir long_term_forecast_traffic_672_96_GPT2WithMM_custom_sl672_ll576_tl96_lr0.0001_bt256_wd1e-05_hd1024_hl2_cosTrue_mixTrue_test_0
done