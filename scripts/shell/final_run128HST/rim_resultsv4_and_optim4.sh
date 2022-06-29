#!/bin/bash
#SBATCH --array=1-300
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-0:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Save_RIM_resultsv4_and_optim
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_results_v4_and_optim.py\
  --experiment_name=medSNR\
  --model=RIMSU128hstv4_augmented_003_K3_L5_BCL2_211124140837_continue_lr6e-05_211129202839\
  --source_model=RIMSource128hstv3_control_009_A1_L3_FLM0.0_211108220845\
  --val_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --train_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --test_dataset=lenses128hst_TNG_rau_200k_control_denoised_testset2_validated\
  --train_size=0\
  --val_size=0\
  --test_size=3000\
  --sie_size=0\
  --buffer_size=1000\
  --lens_coherence_bins=40\
  --source_coherence_bins=40\
  --kappa_coherence_bins=40\
  --seed=42\
  --re_optimize_steps=2000\
  --decay_rate=1
