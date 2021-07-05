#!/bin/bash
#SBATCH --array=1-2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-00:10		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Alpha-Labels-SmokeTest
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_alpha_labels.py\
  --kappa_dir=$HOME/scratch/Censai/data/kappa228_TNG100\
  --output_dir=$HOME/scratch/Censai/data/\
  --compression_type=GZIP\
  --augment=1.\
  --batch=2\
  --bins=10\
  --rescaling_size=100\
  --z_source=2.379\
  --z_lens=0.4457\
  --smoke_test
