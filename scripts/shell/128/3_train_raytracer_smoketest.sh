#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-00:20		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_SmokeTest
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../3_train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha128_NIS\
  --total_items=100\
  --epochs=20\
  --train_split=0.9\
  --compression_type=GZIP\
  --pixels=128\
  --batch_size=10\
  --num_parallel_reads=5\
  --cycle_length=1\
  --block_length=10\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname=RayTracer128_SmokeTest\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=10

