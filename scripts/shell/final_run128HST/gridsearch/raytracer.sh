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
  --datasets $CENSAI_PATH/data/alpha512_TNG100_trainset $CENSAI_PATH/data/alpha512_SIE_trainset\
  --compression_type=GZIP\
  --strategy=uniform\
  --n_models=30\
  --total_items 45000\
  --batch_size 10\
  --epochs=5000\
  --train_split=0.95\
  --pixels=512\
  --initial_learning_rate 1e-4 1e-5\
  --decay_rate 0.5\
  --decay_steps 5000 10000 \
  --kernel_size 5\
  --filters  8 16 32 64\
  --filter_scaling 1\
  --layers 4\
  --block_conv_layers 1 2\
  --strides 4\
  --upsampling_interpolation 1\
  --resampling_kernel_size 5\
  --kappalog\
  --initializer glorot_uniform\
  --block_length=1\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsRT512\
  --logname_prefixe=RT128_512_grid4\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=1\
  --max_to_keep=3\
  --n_residuals=2\
  --patience=50\
  --tolerance=0.01\
  --seed 42 12 10\
  --track_train\
  --source_fov=8\
  --source_w=0.8\
  --psf_sigma=0.06\
  --max_time=23.5
