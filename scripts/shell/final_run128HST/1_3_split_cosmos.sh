#!/bin/bash
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=SplitCosmos
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/split_cosmos.py\
  --dataset=$CENSAI_PATH/data/cosmos_23.5_finalrun128\
  --train_split=0.9\
  --buffer_size=10000
