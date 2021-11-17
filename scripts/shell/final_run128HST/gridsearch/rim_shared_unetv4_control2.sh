#!/bin/bash
#SBATCH --array=1-16
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnetv4_FR128hst_control
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unetv4_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=16\
  --forward_method=fft\
  --epochs=200\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4\
  --decay_rate 1\
  --decay_steps 50000\
  --staircase\
  --patience=80\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --steps 10\
  --flux_lagrange_multiplier 0.\
  --time_weights uniform\
  --kappa_residual_weights sqrt\
  --source_residual_weights uniform\
  --adam 1\
  --rmsprop 0 1\
  --upsampling_interpolation 0\
  --kappalog\
  --source_link identity\
  --filters 32\
  --filter_scaling 1\
  --kernel_size 3\
  --layers 5 6\
  --block_conv_layers 1 2\
  --kernel_size 3\
  --resampling_kernel_size 3\
  --input_kernel_size 11\
  --gru_kernel_size 3\
  --activation tanh bipolar_relu\
  --batch_norm 0\
  --gru_architecture concat\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst3\
  --logname_prefixe=RIMSU128hstv4_control\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42\
  --track_train\
