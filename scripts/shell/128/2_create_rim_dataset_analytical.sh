#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-Analytical-128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../../2_create_rim_dataset_analytical_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128_NIS/\
  --len_dataset=2000\
  --cosmos_dir=$HOME/scratch/Censai/data/cosmos_23.5/\
  --compression_type=GZIP\
  --src_pixels=128\
  --pixels=128\
  --image_fov=10\
  --kappa_fov=15\
  --source_fov=3\
  --noise_rms=0.3e-3\
  --psf_sigma=0.09\
  --max_shift=1.\
  --max_ellipticity=0.6\
  --shuffle_cosmos\
  --buffer_size=1000\
  --tukey_alpha=0.4\
  --batch=20\
  --z_source=2.379\
  --z_lens=0.4457
