#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Plot_RIM_resultsv4
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_plot_samples_v4.py\
  --model_prefix=RIMSU128hstv4_augmented_013_RMSP0_TS8_F32_NLtanh_TWuniform_KRWsqrt_211117182556_continue_lr1e-05_211123105652\
  --val_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --train_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --test_dataset=lenses128hst_TNG_rau_200k_control_denoised_testset_validated\
  --train_size=100\
  --val_size=100\
  --test_size=100\
  --buffer_size=100\
  --batch_size=10\
