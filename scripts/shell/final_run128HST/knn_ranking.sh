#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-0:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=kNN-Ranking
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/knn_vae.py\
  --dataset=$CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated\
  --kappa_vae=$CENSAI_PATH/models/VAE1_128hstfr_002_LS16_dr0.7_betaE0.2_betaDS5000_211115153537\
  --source_dataset=$CENSAI_PATH/data/cosmosFR_VAE1_COSMOSFR_003_F32_NLleaky_relu_LS32_betaE0.1_betaDS100000_211018104400\
  --output_name=k50_vae_ranking_220111\
  -k=50\
  --sample_size=20
