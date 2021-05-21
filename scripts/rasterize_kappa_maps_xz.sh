#!/bin/bash
#SBATCH --array=1-30       # make an array of 10 job like this one to be executed in parallel
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		 # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python rasterize_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/results/kappa612_200kpc\
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_id.npy\
  --projection=xz\
  --pixels=612\
  --fov=0.2\
  --n_neighbors=20
