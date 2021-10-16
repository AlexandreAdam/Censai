#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Kappa-Dataset-900k
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/make_kappa_vae_dataset.py\
  --kappa_first_stage_vae=\
  --output_dir=$CENSAI_PATH/data/kappa128hst_VAE_trainset/\
  --len_dataset=900000\
  --compression_type=GZIP\
  --kappa_fov=7.68\
  --seed=42