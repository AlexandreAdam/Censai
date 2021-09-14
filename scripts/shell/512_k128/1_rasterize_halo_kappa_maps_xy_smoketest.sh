#!/bin/bash 
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00		 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps_512_k228_xy
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rasterize_halo_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/hkappa158_TNG100_512\
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_TNG100-1_id.npy\
  --groupcat_dir=$HOME/projects/rrg-lplevass/data/TNG100-1/groupcat99/\
  --snapshot_dir=$HOME/projects/rrg-lplevass/data/TNG100-1/snapshot99/\
  --offsets=$HOME/projects/rrg-lplevass/data/TNG100-1/offsets/offsets_099.hdf5\
  --snapshot_id=99\
  --projection=xy\
  --pixels=158\
  --fov=0.2\
  --n_neighbors=64\
  --fw_param=3\
  --use_gpu\
  --batch_size=5\
  --smoke_test
