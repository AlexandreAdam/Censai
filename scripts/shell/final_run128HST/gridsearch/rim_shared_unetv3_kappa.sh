#!/bin/bash
#SBATCH --array=1-30
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_KappaUnetv3_wFakeSource_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unetv3_kappa_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_2M_validated_val\
  --compression_type=GZIP\
  --strategy=uniform\
  --n_models=30\
  --forward_method=fft\
  --epochs=200\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4 1e-5\
  --decay_rate 0.9\
  --decay_steps 50000\
  --staircase\
  --patience=10\
  --tolerance=0.01\
  --batch_size 1\
  --train_split=0.9\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --steps 10\
  --time_weights uniform quadratic\
  --kappa_residual_weights uniform sqrt linear\
  --adam 0 1\
  --kappalog\
  --upsampling_interpolation 0\
  --filters 16\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 4 6\
  --block_conv_layers 1\
  --kernel_size 3\
  --resampling_kernel_size 3 6\
  --input_kernel_size 7 11\
  --gru_kernel_size 3\
  --activation leaky_relu tanh\
  --batch_norm 0\
  --gru_architecture concat\
  --kappa_init=0\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst2\
  --logname_prefixe=RIMKappa128hstv3_augmented\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42 314 7\
  --track_train\