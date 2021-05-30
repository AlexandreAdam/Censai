#!/bin/bash
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=CosmosToTFRecords
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python ../cosmos_to_tfrecords.py\
  --img_len=128\
  --example_per_shard=1000\
  --do_all\
  --sample=25.2\
  --exclusion_level=marginal\
  --cosmos_dir=$HOME/scratch/data/COSMOS/COSMOS_25.2_training_sample/\
  --store_attributes\
  --rotation\
  --output_dir=$HOME/scratch/Censai/data/cosmos_25.2/