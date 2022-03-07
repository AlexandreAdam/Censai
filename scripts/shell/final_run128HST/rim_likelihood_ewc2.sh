#!/bin/bash
#SBATCH --array=1-300
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-0:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Optim_RIM_EWC_likelihood
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_reoptimize_regularized_likelihood2.py\
  --experiment_name=optim_ewc2\
  --h5_pattern=optim_ewc*\
  --model=RIMSU128hstv4_augmented_003_K3_L5_BCL2_211124140837_continue_lr6e-05_211129202839\
  --source_vae=VAE1_COSMOSFR_001_F16_NLleaky_relu_LS32_betaE0.1_betaDS100000_220112114306\
  --kappa_vae=VAE1_128hstfr_002_LS16_dr0.7_betaE0.2_betaDS5000_211115153537\
  --dataset=lenses128hst_TNG_rau_200k_control_denoised_testset_validated\
  --sample_size=100\
  --buffer_size=1000\
  --observation_coherence_bins=40\
  --source_coherence_bins=40\
  --kappa_coherence_bins=40\
  --seed=42\
  --learning_rate=5e-7\
  --re_optimize_steps=2000\
  --early_stopping\
  --lam_ewc=0.01\
  --source_vae_ball_size=0.5\
  --kappa_vae_ball_size=0.5
