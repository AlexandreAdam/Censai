#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_VAE_hkappa_Grid_Hparams
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/vae_kappa_gridsearch.py\
  --datasets $CENSAI_PATH/data/hkappa512_TNG100_trainset/\
  --compression_type=GZIP\
  --strategy=uniform\
  --epochs=1000\
  --n_models=10\
  --batch_size 20\
  --train_split=0.9\
  --total_items 50000\
  --optimizer Adam\
  --initial_learning_rate 1e-4 5e-5\
  --decay_rate 0.5\
  --decay_steps=50000\
  --staircase\
  --beta_init 0.1\
  --beta_end_value 0.5 1\
  --beta_decay_power 1\
  --beta_decay_steps 10000 50000\
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
  --layers 4\
  --conv_layers 2\
  --filter_scaling 1\
  --filters 16\
  --kernel_size 5\
  --strides 4\
  --kernel_reg_amp=0\
  --bias_reg_amp=0\
  --activation leaky_relu\
  --batch_norm 0\
  --latent_size 84 256 512\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR\
  --logname_prefixe=VAE1_hk512O\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=2\
  --track_train\
  --max_time=47
