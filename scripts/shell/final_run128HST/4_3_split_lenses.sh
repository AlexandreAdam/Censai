#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-04:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=SplitLense
# 00k-512_k128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/split_lenses.py\
  --dataset=$CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated\
  --train_split=0.9\
  --examples_per_shard=50000\
  --compression_type=GZIP