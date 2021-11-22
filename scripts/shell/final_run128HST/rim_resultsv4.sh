#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=64G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Save_RIM_resultsv4
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/rim_results_v4.py\
  --model_prefix=RIMSU128hstv4\
  --source_model=RIMSource128hstv3_control_001_A0_L2_FLM0.0_211108220845\
  --val_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_val\
  --train_dataset=lenses128hst_TNG_rau_200k_control_denoised_validated_train\
  --test_dataset=
    parser.add_argument("--val_dataset",        required=True,      help="Name of the dataset, not full path")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--train_dataset",      required=True)
    parser.add_argument("--train_size",         default=1000,       type=int)
    parser.add_argument("--val_size",           default=10000,      type=int)
    parser.add_argument("--test_size",          default=10000,      type=int)
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--batch_size",         default=1,          type=int)
    parser.add_argument("--lens_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)