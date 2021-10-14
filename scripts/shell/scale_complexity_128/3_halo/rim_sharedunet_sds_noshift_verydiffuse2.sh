#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_TNGns_128_SDS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128_hTNG100_10k_verydiffuse\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=10\
  --forward_method=fft\
  --epochs=1000\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-3\
  --decay_rate 0.9\
  --decay_steps 5000\
  --staircase\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 10\
  --train_split=0.95\
  --total_items 10000\
  --block_length=1\
  --buffer_size=1000\
  --steps 1 2 3 4 5 6 7 8 9 10\
  --time_weights quadratic\
  --adam 1\
  --upsampling_interpolation 0\
  --kappalog\
  --source_link lrelu4p\
  --filters 32\
  --filter_scaling 1\
  --kernel_size 3\
  --layers 3\
  --block_conv_layers 1\
  --kernel_size 3\
  --resampling_kernel_size 1\
  --input_kernel_size 1\
  --gru_kernel_size 3\
  --activation relu\
  --batch_norm 1\
  --gru_architecture concat\
  --alpha 0.1\
  --source_init=1\
  --kappa_init=0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC2\
  --logname_prefixe=RIMSU128_hTNG2nsvdO_UnrolledSteps2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42\
  --unroll_time_steps\
  --track_train

