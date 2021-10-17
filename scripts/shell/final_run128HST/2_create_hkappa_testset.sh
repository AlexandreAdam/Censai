#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-Kappa-Testset
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/create_kappa_autoencoder_dataset.py\
  --output_dir=$HOME/scratch/Censai/data/hkappa128hst_TNG100_fr_testset/\
  --len_dataset=5000\
  --kappa_dir=$HOME/scratch/Censai/data/hkappa188hst_TNG100\
  --compression_type=GZIP\
  --crop=30\
  --max_shift=0.8\
  --rotate\
  --rotate_by=90\
  --batch=20\
  --bins=10\
  --rescaling_size=100\
  --min_theta_e=1\
  --max_theta_e=3\
  --z_source=2.379\
  --z_lens=0.4457\
  --test_set
