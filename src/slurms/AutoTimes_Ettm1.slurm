#!/bin/bash
#SBATCH --job-name=AutoTimes_Ettm1_replication_GPT2
#SBATCH --partition=spgpu
#SBATCH --account=jjcorso_owned1
#SBATCH --time=00-10:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1     # Running on one node
#SBATCH --ntasks=1    # Running four tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --output=/home/naveenjp/outputs/AutoTimes_Ettm1_replication_GPT2.log
#SBATCH --error=/home/naveenjp/errors/AutoTimes_Ettm1_replication_GPT2.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=naveenjp@umich.edu

model_name=AutoTimes_Gpt2

module load python/3.10.4

# training one model with a context length
model_name=AutoTimes_Gpt2

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_672_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 96 \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --llm_ckp_dir gpt2 \


for test_pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_672_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len $test_pred_len \
  --batch_size 256 \
  --learning_rate 0.0005 \
  --mlp_hidden_layers 0 \
  --train_epochs 10 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --mix_embeds \
  --drop_last \
  --llm_ckp_dir gpt2 \
  --test_dir long_term_forecast_ETTm1_672_96_AutoTimes_Gpt2_ETTm1_sl672_ll576_tl96_lr0.0005_bt256_wd0_hd256_hl0_cosTrue_mixTrue_test_0
done