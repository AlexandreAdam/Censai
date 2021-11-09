#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-04:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Validate_Lenses128hst
# 00k-512_k128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/validate_lenses.py\
  --dataset=$CENSAI_PATH/data/lenses128hst_TNG_VAE_2M\
  --min_magnification=3\
  --signal_threshold=0.1\
  --example_per_worker=100000\
  --compression_type=GZIP\
  --min_source_signal_pixels=10\
  --source_signal_threshold=0.1\
  --edge=5\
  --edge_signal_tolerance=0.2
