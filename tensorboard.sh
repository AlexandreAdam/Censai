#!bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3                     # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=50G                            # memory per node
#SBATCH --time=01:00                         # time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Tensorboard
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
tensorboard --logdir=$CENSAI_PATH/logs --host 0.0.0.0

