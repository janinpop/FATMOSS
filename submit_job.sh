#!/bin/bash
#SBATCH --job-name=fatmoss_gpu
#SBATCH --output=fatmoss_gpu.out
#SBATCH --error=fatmoss_gpu.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=50G

module load cuda
srun python test.py

