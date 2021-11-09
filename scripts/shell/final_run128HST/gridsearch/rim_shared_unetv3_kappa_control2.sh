#!/bin/bash
#SBATCH --array=1-12
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_KappaUnetv3_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unetv3_kappa_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=12\
  --forward_method=fft\
  --epochs=200\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4\
  --decay_rate 0.9\
  --decay_steps 50000\
  --staircase\
  --patience=80\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --steps 8\
  --flux_lagrange_multiplier 0.\
  --time_weights uniform\
  --kappa_residual_weights uniform\
  --adam 1\
  --upsampling_interpolation 0\
  --kappalog\
  --filters 8\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 2 4 6\
  --block_conv_layers 1 2\
  --kernel_size 3\
  --resampling_kernel_size 3\
  --input_kernel_size 11\
  --gru_kernel_size 3\
  --activation leaky_relu\
  --batch_norm 0\
  --gru_architecture concat plus_highway\
  --kappa_init=0\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst2\
  --logname_prefixe=RIMKappa128hstv3_control\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42\
  --track_train\
