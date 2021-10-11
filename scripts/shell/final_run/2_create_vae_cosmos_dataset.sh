#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Create-VAE-Kappa-Dataset-900k
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/make_cosmos_vae_dataset.py\
  --cosmos_first_stage_vae=$CENSAI_PATH/models/VAE1_COSMOS_O_009_L4_CL2_F32_NLbipolar_relu_LS256_betaE0.1_betaDS20000_211011103554\
  --output_dir=$HOME/scratch/Censai/data/cosmosFR_VAE_trainset/\
  --len_dataset=900000\
  --compression_type=GZIP\
  --batch=20\
