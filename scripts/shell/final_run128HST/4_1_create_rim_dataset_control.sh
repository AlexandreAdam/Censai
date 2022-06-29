#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-1
# 00k-512_k128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_rim_dataset.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128hst_TNG_VAE_200k_control/\
  --kappa_datasets $CENSAI_PATH/data/hkappa128hst_TNG100_fr_trainset/\
  --cosmos_datasets $CENSAI_PATH/data/cosmos_23.5_finalrun128_train\
  --compression_type=GZIP\
  --len_dataset=200000\
  --lens_pixels=128\
  --source_fov=3\
  --buffer_size=10000\
  --batch_size=20\
  --tukey_alpha=0\
  --block_length=1\
  --noise_rms_min=0.001\
  --noise_rms_max=0.1\
  --noise_rms_mean=0.01\
  --noise_rms_std=0.03\
  --psf_cutout_size=20\
  --psf_fwhm_min=0.06\
  --psf_fwhm_max=0.3\
  --psf_fwhm_mean=0.08\
  --psf_fwhm_std=0.05

