#!/bin/bash
#SBATCH --job-name=PreprocessETTm1
#SBATCH --partition=spgpu
#SBATCH --account=jjcorso_owned1
#SBATCH --time=00-10:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --output=/home/naveenjp/outputs/preprocess.log
#SBATCH --error=/home/naveenjp/errors/preprocess.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=naveenjp@umich.edu



module load python/3.10.4
source env/bin/activate

python preprocess.py --dataset ETTm1 --gpu 0 --llm_ckp_dir ./gpt2
