#!/bin/bash
#SBATCH --array=1-16
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_SharedUnet_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_shuffleunetv3_gridsearch.py\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_rau_200k_control_validated_val\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=16\
  --forward_method=fft\
  --epochs=200\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-3\
  --decay_rate 0.5\
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
  --steps 8\
  --flux_lagrange_multiplier 1.\
  --time_weights quadratic\
  --kappa_residual_weights uniform linear\
  --source_residual_weights uniform linear\
  --adam 1\
  --kappalog\
  --source_link relu\
  --filters 4 8\
  --kernel_size 3\
  --layers 6\
  --block_conv_layers 1\
  --kernel_size 3\
  --input_kernel_size 1 11\
  --gru_kernel_size 3\
  --activation elu\
  --batch_norm 0\
  --gru_architecture plus\
  --blurpool_kernel_size 5\
  --decoding_blurpool 0\
  --source_init=0\
  --kappa_init=0.1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst\
  --logname_prefixe=RIMSSU128hstv3_control\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=1\
  --n_residuals=2\
  --seed 42\
  --track_train\
