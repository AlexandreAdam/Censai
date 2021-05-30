#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_On_SIE
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python train_rim.py\
  --model_id=21-05-25_15-54-34\
  --epochs=10\
  --total_items=50\
  --batch_size=5\
  --learning_rate=1e-4\
  --decay_rate=1\
  --pixels=64\
  --kappalog=True\
  --adam=True\
  --logdir=../logs\
  --model_dir=../models\
  --checkpoints=1\
  --max_to_keep=1
