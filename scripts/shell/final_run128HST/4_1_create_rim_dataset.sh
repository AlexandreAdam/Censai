#!/bin/bash
#SBATCH --array=1-40
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
python $CENSAI_PATH/scripts/create_rim_dataset_v2.py\
  --output_dir=$HOME/scratch/Censai/data/lenses128hst_TNG_VAE_2M/\
  --kappa_datasets $CENSAI_PATH/data/hkappa128hst_TNG100_fr_trainset/\
 $CENSAI_PATH/data/kappa128hst_VAE1_128hstfr_019_BN1_LS84_betaE0.3_betaDS10000_211018013829/\
  --kappa_datasets_weights 0.4 0.6\
  --cosmos_datasets $CENSAI_PATH/data/cosmos_23.5_finalrun128_train/\
 $CENSAI_PATH/data/cosmosFR_VAE1_COSMOSFR_003_F32_NLleaky_relu_LS32_betaE0.1_betaDS100000_211018104400/\
  --cosmos_datasets_weights 0.4 0.6\
  --compression_type=GZIP\
  --len_dataset=2000000\
  --lens_pixels=128\
  --source_fov=3\
  --noise_rms=0.01\
  --psf_sigma=0.06\
  --buffer_size=50000\
  --batch_size=20\
  --tukey_alpha=0\
  --z_source=2.379\
  --z_lens=0.4457\
  --block_length=1
