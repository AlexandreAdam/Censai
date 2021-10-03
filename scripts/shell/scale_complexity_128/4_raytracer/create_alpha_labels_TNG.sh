#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Alpha-Labels-128-TNG100
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_alpha_labels.py\
  --kappa_dir=$HOME/scratch/Censai/data/kappa158_TNG100_512\
  --len_dataset=15000\
  --output_dir=$HOME/scratch/Censai/data/alpha128_512_TNG\
  --compression_type=GZIP\
  --max_shift=0.5\
  --crop=15\
  --augment\
  --batch=20\
  --bins=10\
  --rescaling_size=100\
  --z_source=2.379\
  --z_lens=0.4457\
