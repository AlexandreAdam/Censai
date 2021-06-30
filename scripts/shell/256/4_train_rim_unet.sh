#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=5 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_On_SIE
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../4_train_rim_unet.py\

