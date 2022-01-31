#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Optim_RIM_horseshoe
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/horseshoe_pred_likelihood_optim_posterior.py\
  --experiment_name=horseshoe10k\
  --rim_data=$CENSAI_PATH/data/horseshoe_rim_data.fits\
  --model=RIMSU128hstv4_augmented_003_K3_L5_BCL2_211124140837_continue_lr6e-05_211129202839\
  --source_vae=VAE1_COSMOSFR_001_F16_NLleaky_relu_LS32_betaE0.1_betaDS100000_220112114306\
  --kappa_vae=VAE1_128hstfr_002_LS16_dr0.7_betaE0.2_betaDS5000_211115153537\
  --reoptimize_step=1000\
  --l2_amp=0\
  --burn_in=800
