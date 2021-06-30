#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer256_NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/3_train_raytracer.py\
  --datasets $HOME/scratch/Censai/data/alpha256_NIS\
  --compression_type=GZIP\
  --total_items=2000\
  --epochs=100\
  --train_split=0.9\
  --compression_type=GZIP\
  --pixels=256\
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
  --batch_size=20\
  --num_parallel_reads=5\
  --cycle_length=1\
  --block_length=20\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RayTracer256_NIS\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --patience=20\
  --tolerance=0.01\
  --seed=42\
