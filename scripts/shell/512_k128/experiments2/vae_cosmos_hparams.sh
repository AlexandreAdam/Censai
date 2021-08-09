#!/bin/bash
#SBATCH --array=1-24
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
  --datasets $CENSAI_PATH/data/cosmos23.5/\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --epochs=200\
  --n_models=24\
  --batch_size 20\
  --train_split=0.9\
  --total_items 40000\
  --optimizer Adam\
  --initial_learning_rate 1e-4\
  --decay_rate 0.5\
  --decay_steps=5000\
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
  --conv_layers_per_block 2 3\
  --filter_scaling 2\
  --filters 32\
  --kernel_size 3\
  --kernel_reg_amp=1e-4\
  --bias_reg_amp=1e-4\
  --activation leaky_relu bipolar_relu\
  --batch_norm 1\
  --latent_size 32 64 96\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsRVAE_k\
  --logname_prefixe=RVAE1_HPARAMS\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=3\
  --track_train\
  --max_time=23.5
