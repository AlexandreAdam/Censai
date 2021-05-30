#!/bin/bash
#SBATCH --array=1-2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_On_SIE
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../train_rim.py\
  --epochs=1000\
  --total_items=200\
  --batch_size=20\
  --learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=2000\
  --staircase\
  --pixels=64\
  --kappalog=True\
  --adam=True\
  --logdir=../../logs\
  --model_dir=../../models\
  --checkpoints=10\
  --max_to_keep=5


