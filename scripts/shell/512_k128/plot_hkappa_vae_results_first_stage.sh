#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=PlotResults_VAE_hk_FirstStage
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/plot_vae_first_stage_results.py\
  --model_prefixe=VAE1_hkappa_HPARAMS2_2_018\
  --compression_type=GZIP\
  --dataset=$CENSAI_PATH/data/hkappa128_TNG100_trainset/\
  --type=kappa\
  --batch_size=10\
  --sampling_size=81\
  --n_plots=5\
  --seed=42
