#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=/work/TALC/enel645_2022w/Software/miniconda3/bin:$PATH
source activate enel645

python /home/drew.burritt/enel645/term-project/train-talc.py
conda deactivate
