#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim_shared_unetv3.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_val\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=1000\
  --max_time=47\
  --optimizer=ADAMAX\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=20000\
  --staircase\
  --clipping\
  --patience=80\
  --tolerance=0.01\
  --batch_size 10\
  --train_split=0.90\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --steps=8\
  --time_weights=uniform\
  --adam\
  --kappalog\
  --source_link=lrelu4p\
  --filters=32\
  --filter_scaling=2\
  --kernel_size=3\
  --layers=4\
  --block_conv_layers=1\
  --kernel_size=3\
  --resampling_kernel_size=3\
  --input_kernel_size=11\
  --gru_kernel_size=3\
  --activation=leaky_relu\
  --gru_architecture=concat\
  --alpha=0.1\
  --source_init=1\
  --kappa_init=0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst\
  --logname_prefixe=RIMSU128hst_controlv3\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=4\
  --n_residuals=2\
  --seed=42\
  --track_train\
  --v2
