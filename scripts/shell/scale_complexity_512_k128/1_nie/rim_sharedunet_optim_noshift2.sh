#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_NIEns_512k128_O2
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k128_NIE_10k\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=24\
  --forward_method=fft\
  --epochs=100000\
  --max_time=47\
  --initial_learning_rate 1e-4\
  --decay_rate 0.9\
  --decay_steps 10000\
  --staircase\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 5\
  --train_split=0.95\
  --total_items 1000\
  --buffer_size=1000\
  --block_length=1\
  --steps 10\
  --adam 0 1\
  --batch_norm 0 1\
  --dropout 0 0.1 0.3\
  --upsampling_interpolation 0\
  --kappalog\
  --source_link lrelu4p\
  --filters 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 4\
  --block_conv_layers 2\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 5\
  --kernel_l2_amp 0\
  --bias_l2_amp 0\
  --kernel_l2_amp 1e-3\
  --bias_l2_amp 1e-3\
  --alpha 0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC2\
  --logname_prefixe=RIMSU512_k128_NIE2nsO2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42 82\
  --track_train

