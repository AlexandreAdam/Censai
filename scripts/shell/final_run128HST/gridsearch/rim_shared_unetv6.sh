#!/bin/bash
#SBATCH --array=1-16
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=4-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnetv6_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unetv6_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_train  $CENSAI_PATH/data/lenses128hst_SIE_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_val  $CENSAI_PATH/data/lenses128hst_SIE_200k_control_validated_val\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=16\
  --forward_method=fft\
  --epochs=500\
  --max_time=95\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4\
  --decay_rate 0.95\
  --decay_steps 100000\
  --staircase\
  --patience=50\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --steps 8\
  --flux_lagrange_multiplier 0.\
  --time_weights uniform\
  --kappa_residual_weights sqrt\
  --source_residual_weights uniform\
  --adam 1\
  --rmsprop 0\
  --upsampling_interpolation 0\
  --kappalog\
  --source_link identity sigmoid\
  --filters 32\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 4 5\
  --block_conv_layers 1 2\
  --kernel_size 3\
  --resampling_kernel_size 3\
  --input_kernel_size 11\
  --gru_kernel_size 3\
  --activation tanh\
  --batch_norm 0\
  --gru_architecture concat plus_highway\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst4\
  --logname_prefixe=RIMSU128hstv6_augmented\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=10\
  --max_to_keep=5\
  --n_residuals=1\
  --seed 42\
  --track_train\
