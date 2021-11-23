#!/bin/bash
#SBATCH --array=1-104%20
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=1-0:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Save_RIM_resultsv4
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_results_v4.py\
  --model_prefix=RIMSU128hstv4\
  --source_model=RIMSource128hstv3_control_001_A0_L2_FLM0.0_211108220845\
  --val_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --train_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --test_dataset=lenses128hst_TNG_rau_200k_control_denoised_testset_validated\
  --train_size=10000\
  --val_size=10000\
  --test_size=30000\
  --buffer_size=10000\
  --batch_size=1\
  --lens_coherence_bins=40\
  --source_coherence_bins=40\
  --kappa_coherence_bins=40