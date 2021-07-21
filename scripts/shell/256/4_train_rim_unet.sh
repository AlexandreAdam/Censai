#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_TNG100_256
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim_unet.py\
  --datasets $HOME/scratch/Censai/data/lenses256_TNG100 $HOME/scratch/Censai/data/lenses256_NIS\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=200\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_steps=10000\
  --staircase\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size=4\
  --train_split=0.9\
  --total_items=500\
  --block_length=1\
  --steps=4\
  --adam\
  --kappalog\
  --kappa_filters=32\
  --kappa_filter_scaling=1\
  --kappa_kernel_size=3\
  --kappa_layers=4\
  --kappa_block_conv_layers=2\
  --kappa_strides=2\
  --kappa_upsampling_interpolation\
  --kappa_kernel_regularizer_amp=1e-4\
  --kappa_bias_regularizer_amp=1e-4\
  --kappa_activatio=leaky_relu\
  --kappa_alpha=0.1\
  --kappa_initializer=glorot_normal\
  --source_filters=32\
  --source_filter_scaling=1\
  --source_kernel_size=3\
  --source_layers=3\
  --source_block_conv_layers=2\
  --source_strides=2\
  --source_upsampling_interpolation\
  --source_kernel_regularizer_amp=1e-4\
  --source_bias_regularizer_amp=1e-4\
  --source_activatio=leaky_relu\
  --source_alpha=0.1\
  --source_initializer=glorot_normal\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RIM_Unet256\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5

