#!/bin/bash
#SBATCH --array=1-4
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=CosmosToTFRecords_Distributed
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/augment_cosmos.py\
    --output_dir=$CENSAI_PATH/data/cosmos_23.5_finalrun512_augmented\
    --batch_size=10\
    --compression_type=GZIP\
    --cosmos_dir=$CENSAI_PATH/data/cosmos_23.5_finalrun612/\
    --crop=50\
    --max_shift=25
