#!/bin/bash
#SBATCH --array=1-72
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_TNG100_512_ScaleFilterSize
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=72\
  --forward_method=fft\
  --epochs=5000\
  --max_time=47\
  --initial_learning_rate=1e-4\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.85\
  --total_items 1000\
  --block_length=1\
  --steps 4 8\
  --adam\
  --kappalog\
  --kappa_filters 8 16 32\
  --kappa_filter_scaling 1 2\
  --kappa_kernel_size 3\
  --kappa_layers 3\
  --kappa_block_conv_layers 3\
  --kappa_strides 4\
  --kappa_upsampling_interpolation\
  --kappa_kernel_regularizer_amp 0\
  --kappa_bias_regularizer_amp 0\
  --kappa_activatio leaky_relu\
  --kappa_alpha 0.1\
  --kappa_initializer glorot_normal\
  --source_filters 8 16 32\
  --source_filter_scaling 1 2\
  --source_kernel_size 3\
  --source_layers 3\
  --source_block_conv_layers 3\
  --source_strides 4\
  --source_upsampling_interpolation\
  --source_kernel_regularizer_amp 0\
  --source_bias_regularizer_amp 0\
  --source_activation leaky_relu\
  --source_alpha 0.1\
  --source_initializer glorot_normal\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logs\
  --logname_prefixe=RIMDU512_SF\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=4\
  --track_train