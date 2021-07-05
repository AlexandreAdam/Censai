#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_TNG100
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha128_TNG100\
  --compression_type=GZIP\
  --total_items=10 100 500 1000 2000 5000 10000 30000 40000\
  --epochs=5000\
  --train_split=0.9\
  --pixels=128\
  --kernel_size=5\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=1000\
  --filters=32\
  --filter_scaling=1\
  --layers=4\
  --block_conv_layers=3\
  --strides=2\
  --resampling_kernel_size=5\
  --kappalog\
  --upsampling_interpolation\
  --kernel_regularizer_amp=0\
  --initializer=glorot_uniform\
  --batch_size=10\
  --num_parallel_reads=10\
  --cycle_length=10\
  --block_length=1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RayTracer128_ScaleDatasetSize\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=5\
  --patience=50\
  --tolerance=0.0\
  --seed=42\
