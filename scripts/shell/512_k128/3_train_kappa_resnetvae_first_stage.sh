#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-02:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_ResnetVAE_FirstStage_TNG100_128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_kappa_resnetvae_first_stage.py\
  --datasets $CENSAI_PATH/data/kappa128_TNG100_trainset/\
  --compression_type=GZIP\
  --epochs=200\
  --initial_learning_rate 1e-4\
  --decay_rate=0.9\
  --decay_steps=1000\
  --beta_init=0\
  --beta_end_value=1.\
  --beta_decay_power=1.\
  --beta_decay_steps=1000\
  --skip_strength_init=1.\
  --skip_strength_end_value=0.\
  --skip_strength_decay_power=0.5\
  --skip_strength_decay_steps=1000\
  --l2_bottleneck_init=1.\
  --l2_bottleneck_end_value=0.\
  --l2_bottleneck_decay_power=0.5\
  --l2_bottleneck_decay_steps=1000\
  --staircase\
  --clipping\
  --patience=20\
  --tolerance=0.01\
  --batch_size=20\
  --train_split=0.9\
  --total_items=100\
  --block_length=1\
  --layers=3\
  --res_blocks_in_layer 4 8 12\
  --conv_layers_per_block=2\
  --filter_scaling=2\
  --filters=32\
  --kernel_size=3\
  --res_architecture=bare\
  --kernel_reg_amp=1e-4\
  --bias_reg_amp=1e-4\
  --activation=bipolar_relu\
  --batch_norm=1\
  --latent_size=16\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsRVAE_k\
  --logname_prefixe=RVAE_1_TNG100\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=10\
