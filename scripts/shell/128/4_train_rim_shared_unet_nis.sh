#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_NIS_128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/4_train_rim_shared_unet.py\
  --datasets $HOME/scratch/Censai/lenses128_NIS\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=200\
  --initial_learning_rate=1e-3\
  --decay_rate=0.9\
  --decay_steps=10000\
  --staircase\
  --clipping=True\
  --patience=20\
  --tolerance=0.01\
  --batch_size=8\
  --train_split=0.9\
  --total_items=1000\
  --num_parallel_reads=4\
  --cycle_length=4\
  --block_length=2\
  --steps=16\
  --adam=True\
  --kappalog=True\
  --kappa_normalize=False\
  --filters=32\
  --filter_scaling=1\
  --kernel_size=3\
  --layers=3\
  --block_conv_layers=2\
  --kernel_size=3\
  --resampling_kernel_size=5\
  --gru_kernel_size=5\
  --kernel_regularizer_amp=1e-4\
  --bias_regularizer_amp=1e-4\
  --alpha=0.1\
  --kappa_resize_filters=8\
  --kappa_resize_method=bilinear\
  --kappa_resize_conv_layers=1\
  --kappa_resize_kernel_size=7\
  --kappa_resize_separate_grad_downsampling=False\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RIM_SharedUnet128_NIS\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5

