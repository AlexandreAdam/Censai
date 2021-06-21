#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G			     # memory per node
#SBATCH --time=0-00:20		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer_SmokeTest
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../3_train_raytracer.py\
  --batch_size=10\
  --total_items=100\
  --logdir=$HOME/scratch/Censai/logs\
  --logname=RayTracer_smoketest\
  --model_dir=$HOME/scratch/Censai/models/\
  --epochs=10\
  --seed=42