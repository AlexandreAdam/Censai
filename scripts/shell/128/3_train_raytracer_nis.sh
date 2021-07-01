#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/3_train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha128_NIS\
  --compression_type=GZIP\
  --total_items=1000\
  --epochs=500\
  --train_split=0.9\
  --pixels=128\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=1000\
  --kernel_size=3\
  --filters=32\
  --filter_scaling=1\
  --layers=4\
  --block_conv_layers=3\
  --strides=2\
  --resampling_kernel_size=5\
  --kappalog=True\
  --normalize=False\
  --upsampling_interpolation=True\
  --kernel_regularizer_amp=0\
  --initializer=glorot_uniform\
  --batch_size=10\
  --num_parallel_reads=5\
  --cycle_length=1\
  --block_length=10\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RayTracer128_NIS\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --patience=10\
  --tolerance=0.01\
  --seed=42
