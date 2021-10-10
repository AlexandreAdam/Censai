#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM), A step takes roughly 2 sec per example with fft
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_DoubleUnet_hTNGnsvd_128_O
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128_TNG100_10k_verydiffuse\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=32\
  --forward_method=fft\
  --epochs=100000\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 5e-4 1e-4\
  --decay_rate 0.5\
  --decay_steps 5000 10000\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 5\
  --train_split=1\
  --total_items 10000\
  --block_length=1\
  --steps 10\
  --time_weights linear quadratic\
  --adam 1\
  --kappalog\
  --delay 0\
  --source_link relu\
  --kappa_filters 16 32\
  --kappa_filter_scaling 2\
  --kappa_kernel_size 3\
  --kappa_layers 4\
  --kappa_block_conv_layers 2\
  --kappa_activation leaky_relu\
  --source_filters 16\
  --source_filter_scaling 2\
  --source_kernel_size 3\
  --source_layers 3\
  --source_block_conv_layers 2\
  --source_activation leaky_relu\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC2\
  --logname_prefixe=RIMDU128_hTNG2nsvdO\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=1\
  --seed 42 142 13\
  --track_train
