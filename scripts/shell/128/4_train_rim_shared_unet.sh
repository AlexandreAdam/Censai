#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_TNG100_128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim_shared_unet.py\
  --datasets $HOME/scratch/Censai/data/lenses128_TNG100 $HOME/scratch/Censai/data/lenses128_NIS\
  --compression_type=GZIP\
  --forward_method=conv2d\
  --epochs=200\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=1000\
  --staircase\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size=4\
  --train_split=0.9\
  --total_items=100\
  --num_parallel_reads=4\
  --cycle_length=4\
  --block_length=1\
  --steps=16\
  --adam\
  --kappalog\
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
  --kappa_resize_separate_grad_downsampling\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$HOME/scratch/Censai/logs\
  --logname_prefixe=RIM_SharedUnet128\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5

