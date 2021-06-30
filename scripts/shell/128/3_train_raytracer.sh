#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_TNG100
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../3_train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha128_TNG100 $HOME/scratch/Censai/data/alpha128_NIS\
  --total_items=1000\
  --epochs=50\
  --train_split=0.9\
  --compression_type=GZIP\
  --pixels=128\
  --kernel_size=3\
  --filters=8\
  --filter_scaling=2\
  --layers=4\
  --block_conv_layers=2\
  --strides=2\
  --resampling_kernel_size=5\
  --kappalog=True\
  --normalize=False\
  --upsampling_interpolation=True\
  --kernel_regularizer_amp=0\
  --initializer=glorot_uniform\
  --batch_size=10\
  --num_parallel_reads=10\
  --cycle_length=10\
  --block_length=1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RayTracer128\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --patience=10\
  --tolerance=0.01\
  --seed=42\