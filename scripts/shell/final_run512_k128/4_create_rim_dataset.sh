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
python $CENSAI_PATH/scripts/create_rim_dataset_v2.py\
  --len_dataset=1000000\
  --output_dir=$HOME/scratch/Censai/data/lenses512_shuffled_200k/\
  --kappa_datasets $CENSAI_PATH/data/hkappa128_TNG100_trainset_fr/ $CENSAI_PATH/data/kappa128_VAE1_hkappa_HPARAMS2_2_018_B20_betaE0.3_betaDS10000_210917123841\
  --kappa_datasets_weights 0.4 0.6\
  --cosmos_datasets $CENSAI_PATH/data/cosmos_23.5_finalrun_train $CENSAI_PATH/data/cosmos_23.5_preprocessed_highSNR_verydiffuse_train/\
  --cosmos_datasets_weights 0.4 0.6\
  --compression_type=GZIP\
  --lens_pixels=512\
  --src_pixels=128\
  --image_fov=17.425909\
  --source_fov=8\
  --noise_rms=0.01\
  --psf_sigma=0.06\
  --buffer_size=5000\
  --batch=20\
  --tukey_alpha=0\
  --z_source=2.379\
  --z_lens=0.4457
