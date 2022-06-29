#!/bin/bash
#SBATCH --array=1-200
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
python $CENSAI_PATH/scripts/create_rim_dataset.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128hst_TNG_VAE_2M/\
  --kappa_datasets $CENSAI_PATH/data/hkappa128hst_TNG100_rau_trainset/\
 $CENSAI_PATH/data/kappa128hst_VAE1_128hstfr_002_LS16_dr0.7_betaE0.2_betaDS5000_211115153537/\
  --kappa_datasets_weights 0.2 0.8\
  --cosmos_datasets $CENSAI_PATH/data/cosmos_23.5_finalrun128_train_denoised\
 $CENSAI_PATH/data/cosmosFR_VAE1_COSMOSFR_003_F32_NLleaky_relu_LS32_betaE0.1_betaDS100000_211018104400/\
  --cosmos_datasets_weights 0.2 0.8\
  --compression_type=GZIP\
  --len_dataset=2000000\
  --lens_pixels=128\
  --source_fov=3\
  --buffer_size=25000\
  --batch_size=20\
  --tukey_alpha=0.\
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
