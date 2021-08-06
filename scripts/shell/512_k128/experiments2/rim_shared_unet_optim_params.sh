#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=4-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_TNG100_512_k128_OPTIM
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses512_k128_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=24\
  --forward_method=fft\
  --epochs=5000\
  --max_time=47\
  --initial_learning_rate 1e-4 5e-5\
  --decay_rate 0.9\
  --decay_steps 200000\
  --optimizer ADAM ADAMAX\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size 5\
  --train_split=0.9\
  --total_items 200000\
  --block_length=1\
  --steps 4\
  --adam\
  --kappalog\
  --source_link sigmoid relu lrelu4p\
  --activation leaky_relu bipolar_leaky_relu\
  --filters 64\
  --filter_scaling 1\
  --kernel_size 3\
  --layers 4\
  --block_conv_layers 3\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 5\
  --kernel_regularizer_amp 1e-4\
  --bias_regularizer_amp 1e-4\
  --alpha 0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsSU512_128\
  --logname_prefixe=RIMSU512_k128_OPTIM\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5\
  --track_train
