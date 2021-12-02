#!/bin/bash
#SBATCH --array=1-100
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_analytic_Gridsearch
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_analytic_gridsearch.py\
  --n_models=100\
  --strategy=uniform\
  --steps 1 2 4 8 12\
  --adam 0 1\
  --layers 2 3 4\
  --units 32 64 128\
  --mlp_before_gru 1 2 3\
  --activation tanh elu leaky_relu\
  --batch_size 1 16 32\
  --total_items 1000\
  --epochs 500\
  --optimizer adamax\
  --initial_learning_rate 1e-3 1e-4\
  --max_time 9.5\
  --checkpoints=10\
  --max_to_keep=1\
  --model_dir=$CENSAI_PATH/models/\
  --logdir=$CENSAI_PATH/logsA/\
  --logname_prefixe=RIMA_g1\
  --seed 42
