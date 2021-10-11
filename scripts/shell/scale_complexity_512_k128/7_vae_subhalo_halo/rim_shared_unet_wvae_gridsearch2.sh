#!/bin/bash
#SBATCH --array=1-10
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
  --model_id=RIMSU512_hk128_TNG2nsO_008_F16_IK7_NLrelu_82_B10_lr0.0005_dr0.5_ds10000_211005114812\
  --strategy=uniform\
  --n_models=10\
  --kappa_first_stage_vae=$CENSAI_PATH/models/VAE1_hkappa_HPARAMS2_2_018_B20_betaE0.3_betaDS10000_210917123841\
  --source_first_stage_vae=$CENSAI_PATH/models/VAE1_COSMOS_O_009_L4_CL2_F32_NLbipolar_relu_LS256_betaE0.1_betaDS20000_211011103554\
  --forward_method=fft\
  --epochs=5000\
  --patience=50\
  --tolerance=0.01\
  --max_time=47\
  --batch_size 5\
  --total_items 2000\
  --image_pixels=512\
  --image_fov=17.425909\
  --kappa_fov=17.425909\
  --source_fov=8\
  --noise_rms=0.05\
  --psf_sigma=0.1\
  --initial_learning_rate 1e-4 1e-5\
  --decay_rate 0.5\
  --decay_steps 10000 50000\
  --optimizer ADAM\
  --clipping\
  --logdir=$CENSAI_PATH/logsSC2\
  --logname_prefixe=RIMSU_wVAE2_pretrained\
  --model_dir=$CENSAI_PATH/models\
  --checkpoints=5\
  --max_to_keep=3\
  --n_residuals=5\
  --seed 1 2 3\
  --json_override=$CENSAI_PATH/models/RIMSU512_hk128_TNG2nsO_008_F16_IK7_NLrelu_82_B10_lr0.0005_dr0.5_ds10000_211005114812/unet_hparams.json
