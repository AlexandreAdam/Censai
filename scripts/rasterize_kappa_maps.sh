#!/bin/bash
#SBATCH --array=1-2       # make an array of 10 job like this one to be executed in parallel
#SBATCH --tasks=1
#SBATCH --cpus-per-task=12 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python rasterize_kappa_maps.py \
  --output_dir=$HOME/scratch/Censai/results/kappa512 \
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_id.npy \
  --projection=xy \
  --pixels=512 \
  --smoke_test