#!/bin/bash
#SBATCH --array=1-2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-1k-128_noisy
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_rim_dataset.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128_hTNG100_1k_verydiffuse_noisy/\
  --len_dataset=1000\
  --kappa_dir=$HOME/scratch/Censai/data/hkappa158_TNG100_512\
  --cosmos_dir=$HOME/scratch/Censai/data/cosmos_23.5_preprocessed_highSNR_verydiffuse/\
  --compression_type=GZIP\
  --lens_pixels=128\
  --src_pixels=128\
  --image_fov=17.425909\
  --source_fov=6\
  --noise_rms=0.1\
  --psf_sigma=0.2\
  --crop=15\
  --max_shift=0\
  --min_theta_e=2\
  --max_theta_e=6\
  --rotate\
  --rotate_by=90\
  --shuffle_cosmos\
  --buffer_size=1000\
  --batch=20\
  --tukey_alpha=0\
  --bins=10\
  --rescaling_size=100\
  --z_source=2.379\
  --z_lens=0.4457
