#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_TNG100_512_k256_ScaleSteps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k256_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=24\
  --forward_method=fft\
  --epochs=5000\
  --max_time=47\
  --initial_learning_rate=1e-4\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 1000\
  --block_length=1\
  --seed 31 42 84 94\
  --steps 4 6 8 10 12 16\
  --adam\
  --kappalog\
  --filters 32\
  --filter_scaling 1\
  --kernel_size 3\
  --layers 5\
  --block_conv_layers 3\
  --kernel_size 5\
  --resampling_kernel_size 7\
  --gru_kernel_size 5\
  --kernel_regularizer_amp 0\
  --bias_regularizer_amp 0\
  --alpha 0.1\
  --kappa_resize_filters 8\
  --kappa_resize_method bilinear\
  --kappa_resize_conv_layers 1\
  --kappa_resize_kernel_size 7\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logs\
  --logname_prefixe=RIMSU512_k256_St\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=1\
  --track_train
