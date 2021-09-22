#!/bin/bash
#SBATCH --array=1-40
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_NIEns_128_O
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128_NIE_10k\
  --compression_type=GZIP\
  --strategy=uniform\
  --n_models=40\
  --forward_method=fft\
  --epochs=100000\
  --max_time=47\
  --initial_learning_rate 1e-4 5e-5 1e-5\
  --decay_rate 1 0.9 0.8 0.5\
  --decay_steps 200 1000 10000\
  --staircase\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 5 10\
  --train_split=0.95\
  --total_items 2000\
  --block_length=1\
  --buffer_size=1000\
  --steps 5 10 15\
  --adam 0 1\
  --upsampling_interpolation 0 1\
  --kappalog\
  --source_link lrelu4p\
  --filters 8 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 3 4 5\
  --block_conv_layers 2 3\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 5\
  --kernel_l2_amp 1e-4\
  --bias_l2_amp 1e-4\
  --alpha 0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC\
  --logname_prefixe=RIMSU128_NIEnsO\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42 82 128\
  --track_train

