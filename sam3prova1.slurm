#!/bin/bash
#SBATCH --job-name=sam3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sam3.out

module load cuda/13.0
module load cudnn/8.7

source ~/sam3_env/bin/activate

python sam3prova1.py