#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-200k
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../2_create_rim_dataset.py\
  --output_dir=$HOME/scratch/aadam/Censai/data/\
  --len_dataset=10\
  --kappa_dir=$HOME/scratch/aadam/Censai/data/kappa612_TNG100_64neighbors\
  --cosmos_dir=$HOME/scratch/aadam/Censai/data/cosmos_23.5/\
  --src_pixels=128\
  --image_fov=20\
  --source_fov=3\
  --noise_rms=0.3e-3\
  --crop=50\
  --shuffle_cosmos\
  --shuffle_buff_size=1000\
  --batch=20\
  --tukey_alpha=0.6\
  --bins=10\
  --rescaling_size=100\
  --z_source=2.379\
  --z_lens=0.4457