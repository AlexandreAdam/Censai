#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=8G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer_SmokeTest
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../3_train_raytracer.py\
  --initializer=random_uniform\
  --decoder_encoder_kernel_size=3\
  --pre_bottleneck_kernel_size=6\
  --bottleneck_kernel_size=16\
  --decoder_encoder_filters=32\
  --filter_scaling=1\
  --upsampling_interpolation=True\
  --kernel_regularizer_amp=1e-4\
  --kappalog=True\
  --normalize=True\
  --batch_size=10\
  --dataset=$HOME/scratch/Censai/data/kappa612_TNG100_64neighbors/\
  --total_items=45360\
  --train_split=0.9\
  --num_parallel_reads=10\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models/\
  --max_to_keep=10\
  --epochs=50\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_steps=1000\
  --clipping=True\
  --patience=10\
  --tolerance=0.01\
  --seed=42