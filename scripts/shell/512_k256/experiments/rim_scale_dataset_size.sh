#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM), A step takes roughly 2 sec per example with fft
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_TNG100_512_k256_ScaleDatasetSize
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k256_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=10\
  --forward_method=fft\
  --epochs=5000\
  --initial_learning_rate=5e-5\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 1 2\
  --train_split=0.85\
  --total_items 100 500 1000 5000 10000\
  --block_length=1\
  --steps 4\
  --adam\
  --kappalog\
  --source_link identity\
  --kappa_filters 16\
  --kappa_filter_scaling 2\
  --kappa_kernel_size 3\
  --kappa_layers 3\
  --kappa_block_conv_layers 2\
  --kappa_strides 2\
  --kappa_upsampling_interpolation\
  --kappa_kernel_regularizer_amp 1e-4\
  --kappa_bias_regularizer_amp 1e-4\
  --kappa_activatio leaky_relu\
  --kappa_alpha 0.1\
  --kappa_initializer glorot_normal\
  --source_filters 16\
  --source_filter_scaling 2\
  --source_kernel_size 3\
  --source_layers 3\
  --source_block_conv_layers 2\
  --source_strides 2\
  --source_upsampling_interpolation\
  --source_kernel_regularizer_amp 1e-4\
  --source_bias_regularizer_amp 1e-4\
  --source_activation leaky_relu\
  --source_alpha 0.1\
  --source_initializer glorot_normal\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logs\
  --logname_prefixe=RIM512_k256_SDS\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=1\
  --track_train