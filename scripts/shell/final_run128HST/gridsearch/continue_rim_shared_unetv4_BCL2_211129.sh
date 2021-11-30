#!/bin/bash
#SBATCH --array=1-8
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=6-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Continue_Train_RIM_SharedUnetv4_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/continue_rim_shared_unetv4.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_train $CENSAI_PATH/data/lenses128hst_SIE_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_val $CENSAI_PATH/data/lenses128hst_SIE_200k_control_validated_val\
  --compression_type=GZIP\
  --models \
  RIMSU128hstv4_augmented_001_K3_L4_BCL2_211124140833\
  RIMSU128hstv4_augmented_003_K3_L5_BCL2_211124140837\
  RIMSU128hstv4_augmented_007_K5_L4_BCL2_211124140833\
  --forward_method=fft\
  --epochs=200\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 6e-5\
  --decay_rate 0.95\
  --decay_steps 100000\
  --patience=40\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst4\
  --logname_prefixe=RIMSU128hstv4\
  --checkpoints=10\
  --max_to_keep=5\
  --n_residuals=1\
  --seed=15\
  --track_train\
  --use_residual_weights