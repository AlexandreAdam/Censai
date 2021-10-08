#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Alpha-Labels-512-TNG100
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_alpha_labels_v2.py\
  --kappa_datasets $CENSAI_PATH/data/kappa512_TNG100_trainset $CENSAI_PATH/data/hkappa512_TNG100_trainset\
  --kappa_datasets_weights 0.5 0.5\
  --output_dir=$HOME/scratch/Censai/data/alpha512_TNG100_trainset\
  --compression_type=GZIP\
  --batch=20\
  --z_source=2.379\
  --z_lens=0.4457\
