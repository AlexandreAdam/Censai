#!/bin/bash
#SBATCH --array=1-30
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=1-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_TNG100_Gridsearch
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/raytracer_gridsearch.py\
  --datasets $CENSAI_PATH/data/alpha128_512_hTNG_15k $CENSAI_PATH/data/alpha128_512_TNG_15k $CENSAI_PATH/data/alpha128_512_SIE_15k\
  --compression_type=GZIP\
  --strategy=uniform\
  --n_models=30\
  --total_items 45000\
  --batch_size 10\
  --epochs=5000\
  --train_split=0.95\
  --pixels=128\
  --initial_learning_rate 1e-3 1e-4\
  --decay_rate 0.9 0.5\
  --decay_steps 10000 50000 100000\
  --kernel_size 3\
  --filters  8 16\
  --filter_scaling 1 2\
  --layers 4\
  --block_conv_layers 2 3\
  --strides 2\
  --resampling_kernel_size 3\
  --kappalog\
  --initializer glorot_uniform\
  --block_length=1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsRT\
  --logname_prefixe=RT128_512_grid2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=1\
  --max_to_keep=3\
  --n_residuals=2\
  --patience=50\
  --tolerance=0.01\
  --seed 42 12 10\
  --track_train\
  --max_time=23.5
