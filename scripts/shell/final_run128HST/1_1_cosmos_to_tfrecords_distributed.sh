#!/bin/bash
#SBATCH --array=1-20
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=CosmosToTFRecords_Distributed
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/cosmos_to_tfrecords_distributed.py\
  --pixels=158\
  --sample=23.5\
  --exclusion_level=marginal\
  --min_flux=50\
  --cosmos_dir=$HOME/projects/rrg-lplevass/data/COSMOS/COSMOS_23.5_training_sample/\
  --output_dir=$HOME/scratch/Censai/data/cosmos_23.5_finalrun158/\
  --preprocess