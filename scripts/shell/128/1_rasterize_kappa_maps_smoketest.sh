#!/bin/bash
#SBATCH --array=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=32G			     # memory per node
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:20		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Rasterize_Kappa_Maps
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rasterize_kappa_maps.py \
  --output_dir=$HOME/scratch/Censai/data \
  --subhalo_id=$HOME/scratch/Censai/data/subhalo_TNG100-1_id.npy\
  --groupcat_dir=$HOME/projects/rrg-lplevass/aadam/data/illustrisTNG100-1_snapshot99_groupcat/\
  --snapshot_dir=$HOME/scratch/data/TNG100-1/snapshot99/\
  --offsets=$HOME/scratch/data/TNG100-1/offsets/offsets_099.hdf5\
  --snapshot_id=99\
  --projection=xy\
  --pixels=228\
  --smoke_test\
  --batch_size=20\
  --use_gpu \
  --smoke_test_id=52623
