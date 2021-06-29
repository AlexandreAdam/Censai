#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8 	# maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:4
#SBATCH --mem=32G		# memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SmokeTest_NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../4_train_rim_unet.py\
  --time_steps=12\
  --adam=True\
  --kappalog=True\
  --normalize=False\
  --kappa_strides=4\
  --source_strides=2\
  --state_size_1=4\
  --state_size_2=32\
  --state_size_3=128\
  --state_size_4=256\
  --forward_method=conv2d\
  --batch_size=4\
  --datasets=$HOME/scratch/Censai/data/lenses_NIS/\
  --compression_type=GZIP\
  --train_split=0.9\
  --total_items=2000\
  --num_parallel_reads=4\
  --block_length=1\
  --cycle_length=4\
  --cache_file=$SLURM_TMPDIR/cache\
  --epochs=10\
  --initial_learning_rate=1e-3\
  --decay_rate=1\
  --staircase\
  --clipping=True\
  --logdir=$HOME/scratch/Censai/logs/\
  --model_dir=$HOME/scratch/Censai/models/\
  --logname=RIM_SmokeTest\
  --checkpoints=1\
  --max_to_keep=1\
  --seed=42
