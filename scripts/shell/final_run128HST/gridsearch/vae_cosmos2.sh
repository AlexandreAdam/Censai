#!/bin/bash
#SBATCH --array=1-64
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_VAECosmos_Grid_Hparams
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/vae_cosmos_gridsearch.py\
  --datasets $CENSAI_PATH/data/cosmos_23.5_finalrun128_train_denoised\
  --compression_type=GZIP\
  --strategy=uniform\
  --epochs=500\
  --n_models=64\
  --batch_size 20\
  --train_split=0.9\
  --total_items 47955\
  --optimizer Adam\
  --initial_learning_rate 1e-4\
  --decay_rate 0.1\
  --decay_steps=200000\
  --beta_init 0.1\
  --beta_end_value 0.1 0.2 0.3 0.4 0.5 0.6 0.9 1.\
  --beta_decay_power 0.5\
  --beta_decay_steps 50000 200000 500000\
  --beta_cyclical 0\
  --skip_strength_init 0\
  --skip_strength_end_value=0.\
  --skip_strength_decay_power 0.5\
  --skip_strength_decay_steps=2000\
  --l2_bottleneck_init=1e-2\
  --l2_bottleneck_end_value=0.\
  --l2_bottleneck_decay_power 0.5\
  --l2_bottleneck_decay_steps=10000\
  --staircase\
  --clipping\
  --block_length=1\
  --layers 3 4 5\
  --conv_layers 2\
  --filter_scaling 1\
  --filters 64 128\
  --kernel_size 3\
  --kernel_reg_amp=0\
  --bias_reg_amp=0\
  --activation leaky_relu\
  --batch_norm 0\
  --latent_size 16 32 64 84 128\
  --cache_file=$SLURM_TMPDIR/cache\
  --logname_prefixe=VAE1_COSMOSFR\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=2\
  --max_to_keep=2\
  --track_train\
  --max_time=23.5
