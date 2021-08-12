#!/bin/bash
#SBATCH --array=1-16
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_VAE2Kappa_Grid_Hparams2
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/vae2_kappa_gridsearch.py\
  --first_stage_model=$CENSAI_PATH/models/VAE1_kappa_HPARAMS_018_L3_CL3_F32_NLbipolar_relu_LS64_ssi0.0_210811184904\
  --datasets $CENSAI_PATH/data/kappa128_TNG100_trainset/\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --epochs=2000\
  --n_models=16\
  --batch_size 20\
  --train_split=0.9\
  --total_items 20000\
  --optimizer Adam\
  --initial_learning_rate 1e-4\
  --decay_rate 0.5\
  --decay_steps=10000\
  --beta_init=0.1\
  --beta_end_value=1\
  --beta_decay_power 1\
  --beta_decay_steps=2000\
  --beta_cyclical 0 1\
  --staircase\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --block_length=1\
  --hidden_layers 2 3\
  --kernel_reg_amp=1e-4\
  --bias_reg_amp=1e-4\
  --activation leaky_relu\
  --latent_size 16 32\
  --units 32 64\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsVAE_kappa\
  --logname_prefixe=VAE2_kappa_HPARAMS\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=3\
  --track_train\
  --max_time=4.7
