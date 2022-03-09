#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=~/Software/miniconda3/bin:$PATH
source activate enel645

python /home/mmylee/enel645/term-project/train-talc.py
conda deactivate
