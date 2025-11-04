#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH -w gnode087

export PYTHONPATH=$(pwd)
echo $PYTHONPATH

cd pretraining
python main.py

cd ../finetuning
python main.py