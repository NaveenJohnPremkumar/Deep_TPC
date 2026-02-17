#!/bin/bash
#SBATCH --job-name=AutoTimes_M4_MSE
#SBATCH --partition=spgpu
#SBATCH --account=jjcorso_owned1
#SBATCH --time=00-10:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1     # Running on one node
#SBATCH --ntasks=1    # Running four tasks
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=48GB
#SBATCH --output=/home/naveenjp/outputs/AutoTimes_M4_MSE.log
#SBATCH --error=/home/naveenjp/errors/AutoTimes_M4_MSE.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=naveenjp@umich.edu

module load python/3.10.4

model_name=AutoTimes_Gpt2
python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model $model_name \
  --data m4 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --loss 'MSE' \
  --use_amp \
  --mlp_hidden_dim 512 \
  --cosine \
  --tmax 10 \
  --llm_ckp_dir ./gpt2 \
  --weight_decay 0.00001

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Quarterly' \
#   --model_id m4_Quarterly \
#   --model $model_name \
#   --data m4 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.00005 \
#   --loss 'SMAPE' \
#   --use_amp \
#   --mlp_hidden_dim 512 \
#   --cosine \
#   --tmax 10 \
#   --llm_ckp_dir ./gpt2 \
#   --weight_decay 0.000005

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Monthly' \
#   --model_id m4_Monthly \
#   --model $model_name \
#   --data m4 \
#   --batch_size 16 \
#   --learning_rate 0.00005 \
#   --des 'Exp' \
#   --itr 1 \
#   --loss 'SMAPE' \
#   --use_amp \
#   --mlp_hidden_dim 1024 \
#   --cosine \
#   --tmax 10 \
#   --llm_ckp_dir ./gpt2 \
#   --weight_decay 0.000001

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Weekly' \
#   --model_id m4_Weekly \
#   --model $model_name \
#   --data m4 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --loss 'SMAPE' \
#   --use_amp \
#   --mlp_hidden_dim 1024 \
#   --cosine \
#   --llm_ckp_dir ./gpt2 \
#   --tmax 10

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id m4_Daily \
#   --model $model_name \
#   --data m4 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0005 \
#   --loss 'SMAPE' \
#   --use_amp \
#   --mlp_hidden_dim 1024 \
#   --llm_ckp_dir ./gpt2 \
#   --weight_decay 0.000005

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Hourly' \
#   --model_id m4_Hourly \
#   --model $model_name \
#   --data m4 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.0001 \
#   --loss 'SMAPE' \
#   --use_amp \
#   --mlp_hidden_dim 1024 \
#   --cosine \
#   --tmax 10 \
#   --llm_ckp_dir ./gpt2 