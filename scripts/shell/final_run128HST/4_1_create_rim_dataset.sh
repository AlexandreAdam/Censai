#!/bin/bash
#SBATCH --array=1-200%17
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
python $CENSAI_PATH/scripts/create_rim_dataset_v2.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128hst_TNG_VAE_2M/\
  --kappa_datasets $CENSAI_PATH/data/hkappa128hst_TNG100_fr_trainset/\
 $CENSAI_PATH/data/kappa128hst_VAE1_128hstfr_019_BN1_LS84_betaE0.3_betaDS10000_211018013829/\
  --kappa_datasets_weights 0.3 0.7\
  --cosmos_datasets $CENSAI_PATH/data/cosmos_23.5_finalrun128_train/\
 $CENSAI_PATH/data/cosmosFR_VAE1_COSMOSFR_003_F32_NLleaky_relu_LS32_betaE0.1_betaDS100000_211018104400/\
  --cosmos_datasets_weights 0.3 0.7\
  --compression_type=GZIP\
  --len_dataset=2000000\
  --lens_pixels=128\
  --source_fov=3\
  --buffer_size=25000\
  --batch_size=20\
  --tukey_alpha=0\
  --block_length=1\
  --noise_rms_min=0.005\
  --noise_rms_max=0.1\
  --noise_rms_mean=0.01\
  --noise_rms_std=0.05\
  --psf_cutout_size=20\
  --psf_fwhm_min=0.06\
  --psf_fwhm_max=0.2\
  --psf_fwhm_mean=0.1\
  --psf_fwhm_std=0.05\
