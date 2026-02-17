#!/bin/bash
#SBATCH --job-name=AutoTimes_Solar
#SBATCH --partition=spgpu
#SBATCH --account=jjcorso_owned1
#SBATCH --time=00-10:00:00
#SBATCH --gpus=4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1     # Running on one node
#SBATCH --ntasks=4    # Running four tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --output=/home/naveenjp/outputs/AutoTimes_Solar.log
#SBATCH --error=/home/naveenjp/errors/AutoTimes_Solar.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=naveenjp@umich.edu


module load python/3.10.4

model_name=AutoTimes_Gpt2

# training one model with a context length
torchrun --nnodes 1 --nproc-per-node 4 run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_672_96 \
  --model $model_name \
  --data Solar \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.000005 \
  --train_epochs 2 \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --des 'Exp' \
  --use_multi_gpu \
  --cosine \
  --tmax 10 \
  --llm_ckp_dir gpt2

# testing the model on all forecast lengths
for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_672_96 \
  --model $model_name \
  --data Solar \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.000005 \
  --train_epochs 2 \
  --use_amp \
  --mlp_hidden_dim 1024 \
  --mlp_activation relu \
  --des 'Exp' \
  --cosine \
  --tmax 10 \
  --llm_ckp_dir gpt2 \
  --test_dir long_term_forecast_solar_672_96_AutoTimes_Gpt2_Solar_sl672_ll576_tl96_lr5e-06_bt256_wd0_hd1024_hl2_cosTrue_mixFalse_Exp_0
done