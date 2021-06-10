#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=6-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_Autoencoder
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../train_cosmos_autoencoder.py\
  --pixels=128\
  --num_parallel_read=3\
  --data=$HOME/scratch/Censai/data/cosmos_25.2\
  --split=0.9\
  --test_shards=2\
  --examples_per_shard=1000\
  --batch_size=20\
  --epochs=50\
  --learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_step=5000\
  --staircase\
  --apodization_alpha=0.1\
  --apodization_factor=1e-3\
  --tv_factor=1e-3\
  --l2_bottleneck=1\
  --l2_bottleneck_decay_steps=5000\
  --l2_bottleneck_decay_power=0.5\
  --skip_strength=0.5\
  --skip_strength_decay_steps=5000\
  --skip_strength_decay_power=0.5\
  --res_architecture="full_pre_activation_rescale"\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models/\
  --max_to_keep=10\
  --checkpoints=5
