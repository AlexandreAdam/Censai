#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM010_FR128hst
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim_shared_unet.py\
  --model_id=RIMSU128hst_control_010_TS10_F16_211020033949\
  --datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_train\
  --val_datasets $CENSAI_PATH/data/lenses128hst_TNG_VAE_200k_control_validated_val\
  --compression_type=GZIP\
  --forward_method=fft\
  --epochs=1000\
  --max_time=47\
  --optimizer ADAMAX\
  --initial_learning_rate 1e-4\
  --decay_rate 0.9\
  --decay_steps 20000\
  --staircase\
  --clipping\
  --patience=80\
  --tolerance=0.01\
  --batch_size 10\
  --train_split=0.90\
  --total_items 10000\
  --block_length=1\
  --buffer_size=10000\
  --cache_file=$SLURM_TMPDIR/cache\
  --logdir=$CENSAI_PATH/logsFR128hst\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=4\
  --n_residuals=2\
  --seed 42\
  --track_train\
  --json_override $CENSAI_PATH/models/RIMSU128hst_control_010_TS10_F16_211020033949/rim_hparams.json\
  $CENSAI_PATH/models/RIMSU128hst_control_010_TS10_F16_211020033949/unet_hparams.json\
  --v2
