#!/bin/bash
#SBATCH --array=1-10
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-01:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-RIM-Dataset-NIE-10k
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_rim_dataset_analytical_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/lenses512_k128_NIE_10k/\
  --len_dataset=10000\
  --cosmos_dir=$HOME/scratch/Censai/data/cosmos_23.5_preprocessed_highSNR/\
  --compression_type=GZIP\
  --lens_pixels=512\
  --src_pixels=128\
  --kappa_pixels=128\
  --image_fov=17.425909\
  --kappa_fov=17.425909\
  --source_fov=10\
  --noise_rms=0.01\
  --psf_sigma=0.08\
  --max_shift=0\
  --min_theta_e=2\
  --max_theta_e=6\
  --max_ellipticity=0.6\
  --shuffle_cosmos\
  --buffer_size=1000\
  --tukey_alpha=0\
  --batch=20\
  --z_source=2.379\
  --z_lens=0.4457
