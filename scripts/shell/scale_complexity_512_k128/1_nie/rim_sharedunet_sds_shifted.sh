#!/bin/bash
#SBATCH --array=1-30
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_NIEs_512k128_SDS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k128_NIE_shifted_10k\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=30\
  --forward_method=fft\
  --epochs=100000\
  --max_time=47\
  --initial_learning_rate 1e-4\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --batch_size 1 5\
  --train_split=1\
  --total_items 100 1000 10000\
  --block_length=1\
  --steps 10\
  --adam 1\
  --kappalog\
  --source_link relu\
  --filters 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 4\
  --block_conv_layers 2\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 5\
  --kernel_l2_amp 1e-4\
  --bias_l2_amp 1e-4\
  --alpha 0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC\
  --logname_prefixe=RIMSU512_k128_NIEs2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=1\
  --seed 2 4 8 16 32\
  --track_train

