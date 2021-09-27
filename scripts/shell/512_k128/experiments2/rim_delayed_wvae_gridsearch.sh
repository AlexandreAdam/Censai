#!/bin/bash
#SBATCH --array=1-8
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM), A step takes roughly 2 sec per example with fft
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIM_TNG100_512_k128_wVAE
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_delayed_wvae_gridsearch.py\
  --kappa_first_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741\
  --kappa_second_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741_second_stage_210813104442\
  --source_first_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842\
  --source_second_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842_second_stage_210812235647\
  --strategy=exhaustive\
  --n_models=8\
  --forward_method=fft\
  --epochs=5000\
  --max_time=47\
  --batch_size 1\
  --total_items 2000\
  --image_pixels=512\
  --image_fov=20\
  --kappa_fov=20\
  --source_fov=3\
  --noise_rms=5e-2\
  --psf_sigma=0.1\
  --initial_learning_rate=5e-5\
  --clipping\
  --patience=40\
  --tolerance=0.01\
  --steps 10\
  --adam 0 1\
  --kappalog\
  --delay 1 4 8 16\
  --source_link relu\
  --kappa_filters 16\
  --kappa_filter_scaling 2\
  --kappa_kernel_size 3\
  --kappa_layers 4\
  --kappa_block_conv_layers 2\
  --kappa_strides 2\
  --kappa_activation leaky_relu\
  --kappa_alpha 0.1\
  --kappa_initializer glorot_normal\
  --source_filters 16\
  --source_filter_scaling 2\
  --source_kernel_size 3\
  --source_layers 4\
  --source_block_conv_layers 2\
  --source_strides 2\
  --source_activation leaky_relu\
  --source_alpha 0.1\
  --source_initializer glorot_normal\
  --logdir=$CENSAI_PATH/logsRIMDU_D_wVAE\
  --logname_prefixe=RIMDU_D_wVAE\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=2
