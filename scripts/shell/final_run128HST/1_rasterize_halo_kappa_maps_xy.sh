#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=64G			     # memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00		 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Rasterize_hKappa_Maps_188_xy
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rasterize_halo_kappa_maps.py\
  --output_dir=$HOME/scratch/Censai/data/hkappa188hst_TNG100\
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_TNG100-1_id.npy\
  --groupcat_dir=$HOME/projects/rrg-lplevass/data/TNG100-1/groupcat99/\
  --snapshot_dir=$HOME/projects/rrg-lplevass/data/TNG100-1/snapshot99/\
  --offsets=$HOME/projects/rrg-lplevass/data/TNG100-1/offsets/offsets_099.hdf5\
  --snapshot_id=99\
  --projection=xy\
  --pixels=188\
  --fov=0.105\
  --n_neighbors=64\
  --fw_param=3\
  --use_gpu\
  --batch_size=5