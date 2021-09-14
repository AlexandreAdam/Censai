#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-08:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_ResnetVAE_kappa_Grid_Optim
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/resnet_vae_kappa_gridsearch.py\
  --datasets $CENSAI_PATH/data/kappa128_TNG100_trainset/\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --epochs=200\
  --n_models=24\
  --batch_size 20\
  --train_split=0.9\
  --total_items 20000\
  --optimizer Adam Adamax\
  --initial_learning_rate 1e-2 1e-3 1e-4\
  --decay_rate 1 0.9\
  --decay_steps=2000\
  --beta_init=0\
  --beta_end_value=1.\
  --beta_decay_power 1.\
  --beta_decay_steps=2000\
  --beta_cyclical 0 1\
  --skip_strength_init=0.\
  --skip_strength_end_value=0.\
  --skip_strength_decay_power 0.5\
  --skip_strength_decay_steps=2000\
  --l2_bottleneck_init=1.\
  --l2_bottleneck_end_value=0.\
  --l2_bottleneck_decay_power 0.5\
  --l2_bottleneck_decay_steps=2000\
  --staircase\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --block_length=1\
  --layers 3\
  --res_blocks_in_layer 4\
  --conv_layers_per_block 2\
  --filter_scaling 2\
  --filters 32\
  --kernel_size 3\
  --res_architecture full_pre_activation\
  --kernel_reg_amp=1e-4\
  --bias_reg_amp=1e-4\
  --activation relu leaky_relu\
  --batch_norm 1\
  --latent_size 84\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsRVAE_k\
  --logname_prefixe=RVAE1_kap_OPTIM_1\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=3\
  --track_train\
  --max_time=7.7
