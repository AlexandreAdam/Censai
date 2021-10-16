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
  --output_dir=$HOME/scratch/Censai/data/lenses512_shuffled_200k/\
  --kappa_datasets $CENSAI_PATH/data/kappa512_TNG100_trainset_fr $CENSAI_PATH/data/hkappa512_TNG100_trainset_fr\
  --compression_type=GZIP\
  --lens_pixels=512\
  --src_pixels=128\
  --image_fov=17.425909\
  --source_fov=10\
  --noise_rms=0.01\
  --psf_sigma=0.06\
  --crop=15\
  --max_shift=1\
  --min_theta_e=1.5\
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
