#!/bin/bash
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3  # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-02:00		 # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Validate_Kappa_Maps_128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/validate_kappa_maps.py\
  --kappa_dir=$HOME/scratch/Censai/data/hkappa612_TNG100_512\
  --good_kappa_cutoff=1\
  --train_split=0.9
