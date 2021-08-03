#!/bin/bash
#SBATCH --array=1-15
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_TNG100_128_ScaleDatasetSize
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=15\
  --forward_method=fft\
  --epochs=5000\
  --max_time=47\
  --initial_learning_rate 5e-5\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 1 5 10\
  --train_split=0.9\
  --total_items 100 500 1000 5000 10000\
  --block_length=1\
  --steps 4\
  --adam\
  --kappalog\
  --source_link linear\
  --filters 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 3\
  --block_conv_layers 2\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 5\
  --kernel_regularizer_amp 1e-4\
  --bias_regularizer_amp 1e-4\
  --alpha 0.1\
  --kappa_resize_filters 8\
  --kappa_resize_method bilinear\
  --kappa_resize_conv_layers 1\
  --kappa_resize_kernel_size 7\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logs\
  --logname_prefixe=RIMSU128_SDS\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --track_train

