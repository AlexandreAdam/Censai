#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-1
# 00k-512_k128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_rim_dataset_analytical_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128hst_SIE_200k_control/\
  --cosmos_dir=$CENSAI_PATH/data/cosmos_23.5_finalrun128_train\
  --compression_type=GZIP\
  --len_dataset=200000\
  --lens_pixels=128\
  --kappa_pixels=128\
  --src_pixels=128\
  --image_fov=7.68\
  --kappa_fov=7.68\
  --source_fov=3\
  --buffer_size=50000\
  --batch_size=20\
  --tukey_alpha=0.1\
  --block_length=1\
  --max_shift=1.\
  --min_theta_e=1.\
  --max_theta_e=3.\
  --shuffle_cosmos\
  --buffer_size=50000\
  --noise_rms_min=0.001\
  --noise_rms_max=0.1\
  --noise_rms_mean=0.01\
  --noise_rms_std=0.05\
  --psf_cutout_size=20\
  --psf_fwhm_min=0.08\
  --psf_fwhm_max=0.5\
  --psf_fwhm_mean=0.2\
  --psf_fwhm_std=0.1
