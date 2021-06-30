#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 	# maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G		# memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_NIS_256
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../4_train_rim_unet.py\
  --datasets $HOME/scratch/Censai/data/lenses256_NIS\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=200\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_steps=5000\
  --staircase\
  --clipping=True\
  --patience=20\
  --tolerance=0.01\
  --batch_size=8\
  --train_split=0.9\
  --total_items=5000\
  --num_parallel_reads=4\
  --cycle_length=4\
  --block_length=2\
  --steps=16\
  --adam=True\
  --kappalog=True\
  --kappa_normalize=False\
  --kappa_filters=32\
  --kappa_filter_scaling=1\
  --kappa_kernel_size=3\
  --kappa_layers=4\
  --kappa_block_conv_layers=2\
  --kappa_strides=2\
  --kappa_upsampling_interpolation=False\
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
  --source_upsampling_interpolation=False\
  --source_kernel_regularizer_amp=1e-4\
  --source_bias_regularizer_amp=1e-4\
  --source_activatio=leaky_relu\
  --source_alpha=0.1\
  --source_initializer=glorot_normal\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RIM_Unet256_NIS\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5


