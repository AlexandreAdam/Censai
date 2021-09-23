#!/bin/bash
#SBATCH --array=1-32
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_NIEns_512k128_SDS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k128_NIE_10k\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=32\
  --forward_method=fft\
  --epochs=1000\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4\
  --decay_rate 0.8\
  --decay_steps 1000\
  --clipping\
  --patience=100\
  --tolerance=0.01\
  --batch_size 5\
  --train_split=1\
  --total_items 5 10 100 1000\
  --block_length=1\
  --buffer_size=1000\
  --steps 10\
  --time_weights linear quadratic\
  --adam 1\
  --kappalog\
  --source_link relu\
  --filters 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 4\
  --block_conv_layers 2\
  --kernel_size 3\
  --resampling_kernel_size 3\
  --input_kernel_size 7\
  --gru_kernel_size 3\
  --activation relu\
  --gru_architecture plus\
  --alpha 0.3\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSC2\
  --logname_prefixe=RIMSU512_k128_NIE2ns\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=1\
  --seed 2 4 8 16\
  --track_train

