#!/bin/bash
#SBATCH --array=1-2
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-00:10		# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python rasterize_kappa_maps.py \
  --output_dir=$HOME/scratch/Censai/results \
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_id.npy \
  --projection=xy \
  --pixels=612 \
  --smoke_test
