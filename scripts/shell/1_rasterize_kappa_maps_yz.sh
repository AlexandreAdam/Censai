#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=6-00:00		 # time (DD-HH:MM)
#SBATCH --account=def-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../1_rasterize_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/kappa612_TNG100_64neighbors\
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_TNG100-1_id.npy\
  --projection=yz\
  --pixels=612\
  --fov=0.2\
  --n_neighbors=64\
  --fw_param=3
