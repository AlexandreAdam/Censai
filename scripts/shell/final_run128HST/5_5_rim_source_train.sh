#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIMSource_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_val\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=100\
  --max_time=23\
  --optimizer ADAM\
  --initial_learning_rate 1e-4\
  --decay_rate 0.9\
  --decay_steps 50000\
  --staircase\
  --clipping\
  --patience=80\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.90\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --filters=32\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst\
  --logname_prefixe=RIMSource128hst_f32\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=4\
  --n_residuals=2\
  --seed 42\
  --track_train\
  --v2
