#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer_NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../3_train_raytracer512.py\
  --initializer=glorot_uniform\
  --decoder_encoder_kernel_size=3\
  --pre_bottleneck_kernel_size=6\
  --bottleneck_strides=4\
  --bottleneck_kernel_size=16\
  --decoder_encoder_filters=8\
  --filter_scaling=2\
  --upsampling_interpolation=True\
  --kernel_regularizer_amp=0\
  --kappalog=True\
  --normalize=False\
  --datasets $HOME/scratch/Censai/data/alpha512_NIS/\
  --compression_type=GZIP\
  --total_items=5000\
  --train_split=0.9\
  --batch_size=10\
  --num_parallel_reads=5\
  --cycle_length=1\
  --block_length=10\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --model_dir=$HOME/scratch/Censai/models/\
  --max_to_keep=10\
  --checkpoints=5\
  --epochs=100\
  --initial_learning_rate=1e-3\
  --decay_rate=1\
  --decay_steps=1000\
  --clipping=False\
  --patience=15\
  --tolerance=0.\
  --seed=42
