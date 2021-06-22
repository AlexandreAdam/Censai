#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=5 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:2
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SmokeTest_NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../4_train_rim_unet.py\
  --epochs=10\
  --total_items=50\
  --batch_size=5\
  --initial_learning_rate=1e-4\
  --decay_rate=1\
  --pixels=64\
  --kappalog=True\
  --adam=True\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=1\
  --max_to_keep=1\
  --logname="RIM_smoketest_NIS"
