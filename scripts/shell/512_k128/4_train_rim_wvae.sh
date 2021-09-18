#!/bin/bash
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=0-05:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_TNG100_128
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/train_rim_unet_vae_dataset.py\
  --kappa_first_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741\
  --kappa_second_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741_second_stage_210813104442\
  --source_first_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842\
  --source_second_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842_second_stage_210812235647\
  --forward_method=fft\
  --epochs=50\
  --max_time=47\
  --batch_size 1\
  --total_items 200\
  --image_pixels=512\
  --image_fov=20\
  --kappa_fov=20\
  --source_fov=3\
  --noise_rms=5e-2\
  --psf_sigma=0.1\
  --initial_learning_rate=1e-4\
  --decay_rate=0.9\
  --decay_steps=10000\
  --staircase\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --steps=4\
  --adam\
  --kappalog\
  --kappa_filters=32\
  --kappa_filter_scaling=1\
  --kappa_kernel_size=3\
  --kappa_layers=3\
  --kappa_block_conv_layers=2\
  --kappa_strides=2\
  --kappa_upsampling_interpolation\
  --kappa_kernel_regularizer_amp=1e-4\
  --kappa_bias_regularizer_amp=1e-4\
  --kappa_activation=leaky_relu\
  --kappa_alpha=0.1\
  --kappa_initializer=glorot_normal\
  --source_filters=32\
  --source_filter_scaling=1\
  --source_kernel_size=3\
  --source_layers=3\
  --source_block_conv_layers=2\
  --source_strides=2\
  --source_upsampling_interpolation\
  --source_kernel_regularizer_amp=1e-4\
  --source_bias_regularizer_amp=1e-4\
  --source_activatio=leaky_relu\
  --source_alpha=0.1\
  --source_initializer=glorot_normal\
  --logdir=$HOME/scratch/Censai/logsRIMDU_wVAE\
  --logname_prefixe=RIMDU_wVAE\
  --model_dir=$HOME/scratch/Censai/models\
  --checkpoints=5\
  --max_to_keep=10\
  --n_residuals=5


