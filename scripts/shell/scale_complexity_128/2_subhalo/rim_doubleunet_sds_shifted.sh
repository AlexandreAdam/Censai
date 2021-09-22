#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM), A step takes roughly 2 sec per example with fft
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_DoubleUnet_TNGns_128_SDS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128_TNG100_shifted_10k\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=24\
  --forward_method=fft\
  --epochs=100000\
  --max_time=47\
  --initial_learning_rate=1e-4\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 1 2\
  --train_split=1\
  --total_items 100 1000 10000\
  --block_length=1\
  --buffer_size=1000\
  --steps 10\
  --adam 1\
  --kappalog\
  --delay 0 5\
  --source_link relu\
  --kappa_filters 16\
  --kappa_filter_scaling 2\
  --kappa_kernel_size 3\
  --kappa_layers 4\
  --kappa_block_conv_layers 2\
  --kappa_strides 2\
  --kappa_upsampling_interpolation\
  --kappa_kernel_regularizer_amp 0\
  --kappa_bias_regularizer_amp 0\
  --kappa_activation leaky_relu\
  --kappa_alpha 0.1\
  --kappa_initializer glorot_normal\
  --source_filters 16\
  --source_filter_scaling 2\
  --source_kernel_size 3\
  --source_layers 3\
  --source_block_conv_layers 2\
  --source_strides 2\
  --source_upsampling_interpolation\
  --source_kernel_regularizer_amp 0\
  --source_bias_regularizer_amp 0\
  --source_activation leaky_relu\
  --source_alpha 0.1\
  --source_initializer glorot_normal\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC\
  --logname_prefixe=RIMDU512_k128_TNGs\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=1\
  --seed 42 142\
  --track_train