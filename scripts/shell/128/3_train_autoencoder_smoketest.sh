#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_Autoencoder
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../3_train_cosmos_autoencoder.py\
  --pixels=128\
  --num_parallel_read=3\
  --data=$HOME/scratch/Censai/data/cosmos_25.2\
  --split=0.95\
  --test_shards=50\
  --examples_per_shard=1000\
  --batch_size=100\
  --epochs=1\
  --learning_rate=1e-4\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models/\
  --logname="cosmosAE_smoketest"\
