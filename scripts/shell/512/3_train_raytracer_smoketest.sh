#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-00:20		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer_SmokeTest
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../3_train_raytracer512.py\
  --initializer=glorot_uniform\
  --decoder_encoder_kernel_size=3\
  --pre_bottleneck_kernel_size=6\
  --bottleneck_strides=4\
  --bottleneck_kernel_size=16\
  --decoder_encoder_filters=32\
  --filter_scaling=1\
  --upsampling_interpolation=True\
  --kernel_regularizer_amp=1e-4\
  --kappalog=True\
  --normalize=False\
  --datasets $HOME/scratch/Censai/data/alpha512_NIS/ $HOME/scratch/Censai/data/alpha512_TNG100\
  --compression_type=GZIP\
  --total_items=500\
  --train_split=0.9\
  --batch_size=16\
  --num_parallel_reads=4\
  --cycle_length=4\
  --block_length=4\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models/\
  --max_to_keep=10\
  --n_residuals=10\
  --checkpoints=1\
  --epochs=10\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_steps=1000\
  --clipping=True\
  --patience=10\
  --tolerance=0.01\
  --seed=42