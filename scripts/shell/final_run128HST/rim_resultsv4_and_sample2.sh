#!/bin/bash
#SBATCH --array=1-50
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-0:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Save_RIM_resultsv4_and_sample
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_results_v4_and_sample.py\
  --experiment_name=slgd_sample_medSNR\
  --model=RIMSU128hstv4_augmented_003_K3_L5_BCL2_211124140837_continue_lr6e-05_211129202839\
  --val_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --train_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --test_dataset=lenses128hst_TNG_rau_200k_control_denoised_testset2_validated\
  --train_size=0\
  --val_size=0\
  --test_size=100\
  --sie_size=0\
  --buffer_size=1000\
  --lens_coherence_bins=40\
  --source_coherence_bins=40\
  --kappa_coherence_bins=40\
  --seed=42\
  --burn_in=1000\
  --sampling_steps=3000\
  --learning_rate=1e-6\
  --decay_steps=500\
  --decay_rate=1
