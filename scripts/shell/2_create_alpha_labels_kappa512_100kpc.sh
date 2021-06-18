#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Alpha-Labels-Kappa512_100kpc
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../2_create_alpha_labels.py \
  --kappa_dir=$HOME/scratch/Censai/data/kappa512_100kpc\
  --output_dir=$HOME/scratch/Censai/data/alpha512_100kpc\
  --image_fov=20\
  --augment=1.\
  --batch=10\
  --bins=10\
  --rescaling_size=100\
  --z_source=2.379\
  --z_lens=0.4457\
