#!/bin/bash
#SBATCH --array=1-24
#SBATCH --tasks=1
#SBATCH --cpus-per-task=3 # maximum cpu per task is 3.5 per gpus
#SBATCH --gres=gpu:1
#SBATCH --mem=32G			     # memory per node
#SBATCH --time=2-00:00		# time (DD-HH:MM)
#SBATCH --account=rrg-lplevass
#SBATCH --job-name=Train_RIMSU_wVAE_Gridsearch2
#SBATCH --output=%x-%j.out
source $HOME/environments/censai3.8/bin/activate
python $CENSAI_PATH/scripts/experiments/rim_shared_unet_wvae_gridsearch.py\
  --strategy=exhaustive\
  --n_models=24\
  --kappa_first_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741\
  --kappa_second_stage_vae=$CENSAI_PATH/models/VAE1_kappa_HPARAMS2_010_CL2_F64_NLbipolar_relu_LS32_210812184741_second_stage_210813104442\
  --source_first_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842\
  --source_second_stage_vae=$CENSAI_PATH/models/VAE1_cosmos_HPARAMS_029_L3_CL4_F16_NLleaky_relu_LS128_ssi0.001_210810161842_second_stage_210812235647\
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
  --initial_learning_rate 1e-4\
  --decay_rate 0.5\
  --decay_steps 200000 100000\
  --optimizer ADAM\
  --clipping\
  --steps 2 4\
  --adam 1\
  --kappalog\
  --source_link relu\
  --activation leaky_relu bipolar_leaky_relu\
  --filters 32\
  --filter_scaling 2\
  --kernel_size 3\
  --layers 3\
  --block_conv_layers 2 3 4\
  --kernel_size 3\
  --resampling_kernel_size 5\
  --gru_kernel_size 3\
  --kernel_regularizer_amp 1e-4\
  --bias_regularizer_amp 1e-4\
  --logdir=$CENSAI_PATH/logsSU_wVAE2\
  --logname_prefixe=RIMSU_wVAE2\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=5
