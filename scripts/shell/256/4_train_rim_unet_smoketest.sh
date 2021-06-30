#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SmokeTest
#SBATCH --output=%x-%j.out
#SBATCH --job-name=Train_RIM_SmokeTest_256
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../4_train_rim_unet.py\
  --datasets $HOME/scratch/Censai/data/lenses256_NIS\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=20\
  --initial_learning_rate=1e-3\
  --batch_size=8\
  --train_split=0.9\
  --total_items=100\
  --num_parallel_reads=4\
  --cycle_length=4\
  --block_length=2\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname=RIM_Unet256_SmokeTest\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5

