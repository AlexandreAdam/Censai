#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_VAE_hkappa_Grid_Hparams2
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/vae_kappa_gridsearch.py\
  --datasets $CENSAI_PATH/data/hkappa128_TNG100_trainset/\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --epochs=1000\
  --n_models=24\
  --batch_size 10 20\
  --train_split=0.9\
  --total_items 20000\
  --optimizer Adam\
  --initial_learning_rate 1e-4\
  --decay_rate 0.5\
  --decay_steps=50000\
  --staircase\
  --beta_init 0 0.1\
  --beta_end_value 0.1 0.2 1\
  --beta_decay_power 1\
  --beta_decay_steps=50000\
  --beta_cyclical 0\
  --skip_strength_init 0\
  --skip_strength_end_value=0.\
  --skip_strength_decay_power 0.5\
  --skip_strength_decay_steps=2000\
  --l2_bottleneck_init=1e-2\
  --l2_bottleneck_end_value=0.\
  --l2_bottleneck_decay_power 0.5\
  --l2_bottleneck_decay_steps=10000\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --block_length=1\
  --layers 3\
  --conv_layers 3\
  --filter_scaling 2\
  --filters 16\
  --kernel_size 3\
  --kernel_reg_amp=1e-4\
  --bias_reg_amp=1e-4\
  --activation leaky_relu\
  --batch_norm 0 1\
  --latent_size 84\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsVAE_hkappa\
  --logname_prefixe=VAE1_hkappa_HPARAMS2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=0\
  --track_train\
  --max_time=47
