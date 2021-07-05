#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer256_TNG100
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha256_TNG100 $HOME/scratch/Censai/data/alpha256_NIS\
  --compression_type=GZIP\
  --total_items=5000\
  --epochs=500\
  --train_split=0.9\
  --compression_type=GZIP\
  --pixels=256\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=1000\
  --kernel_size=3\
  --filters=8\
  --filter_scaling=2\
  --layers=4\
  --block_conv_layers=2\
  --strides=2\
  --resampling_kernel_size=5\
  --kappalog\
  --upsampling_interpolation\
  --kernel_regularizer_amp=1e-5\
  --initializer=glorot_uniform\
  --batch_size=20\
  --num_parallel_reads=5\
  --cycle_length=5\
  --block_length=4\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RayTracer256\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --patience=50\
  --tolerance=0.01\
  --seed=42\
