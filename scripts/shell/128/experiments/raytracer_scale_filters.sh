#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=16G			     # memory per node
#SBATCH --time=0-10:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RayTracer128_TNG100_ScaleFilters
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/raytracer_gridsearch.py\
  --datasets $CENSAI_PATH/data/alpha128_TNG100\
  --compression_type=GZIP\
  --strategy=exhaustive\
  --n_models=24\
  --total_items 100 1000 5000\
  --epochs=5000\
  --train_split=0.9\
  --pixels=128\
  --initial_learning_rate=1e-4\
  --kernel_size 5\
  --filters 8 16 32 64\
  --filter_scaling 1 2\
  --layers 4\
  --block_conv_layers 3\
  --strides 2\
  --resampling_kernel_size 5\
  --kappalog\
  --upsampling_interpolation\
  --kernel_regularizer_amp 0\
  --initializer glorot_uniform\
  --batch_size=10\
  --num_parallel_reads=10\
  --cycle_length=10\
  --block_length=1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logs\
  --logname_prefixe=RT128_SF\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=5\
  --patience=50\
  --tolerance=0.0\
  --seed=42\
  --track_train