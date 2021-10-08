#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-00:30		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Kappa-Dataset-20k
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_kappa_autoencoder_dataset.py\
  --output_dir=$HOME/scratch/Censai/data/kappa512_TNG100_trainset/\
  --len_dataset=20000\
  --kappa_dir=$HOME/scratch/Censai/data/kappa612_TNG100\
  --compression_type=GZIP\
  --crop=50\
  --max_shift=1.\
  --rotate\
  --rotate_by=90\
  --batch=20\
  --bins=10\
  --rescaling_size=100\
  --min_theta_e=2\
  --max_theta_e=6\
  --z_source=2.379\
  --z_lens=0.4457
