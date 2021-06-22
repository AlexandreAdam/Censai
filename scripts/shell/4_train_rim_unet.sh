#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=5 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:2
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_On_SIE
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../4_train_rim.py\
  --time_steps=16\
  --adam=True\
  --logkappa=True\
  --normalize=False\
  --kappa_strides=4\
  --source_strides=2\
  --forward_method=fft\
  --batch_size=10\
  --dataset=$HOME/scratch/Censai/data/lenses_TNG100/\
  --train_split=0.9\
  --total_items=200000\
  --num_parallel_reads=5\
  --cache_file=$SLURM_TMPDIR/cache\
  --epochs=50\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_step=20000\
  --staircase\
  --clipping=True\
  --logdir=$HOME/scratch/Censai/logs/\
  --model_dir=$HOME/scratch/Censai/models/\
  --logname=RIM_SmokeTest\
  --checkpoints=1\
  --max_to_keep=1\
  --seed=42