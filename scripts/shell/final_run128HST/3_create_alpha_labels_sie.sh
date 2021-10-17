#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-1:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Alpha-Labels-128-NIS
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_alpha_labels_analytical_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/alpha512_SIE_trainset\
  --pixels=512\
  --len_dataset=15000\
  --compression_type=GZIP\
  --batch=20\
  --max_shift=1\
  --image_fov=17.99538\
  --kappa_fov=17.99538\
  --min_theta_e=2\
  --max_theta_e=6\
  --max_ellipticity=0.6\
  --z_source=2.379\
  --z_lens=0.4457\
