#!/bin/bash
python $CENSAI_PATH/scripts/knn_vae.py\
  --dataset=$CENSAI_PATH2/data/lenses128hst_TNG_rau_200k_control\
  --kappa_vae=$CENSAI_PATH2/models/VAE1_128hstfr_002_LS16_dr0.7_betaE0.2_betaDS5000_211115153537\
  --source_vae=$CENSAI_PATH2/models/VAE1_COSMOSFR_001_F16_NLleaky_relu_LS32_betaE0.1_betaDS100000_220112114306\
  --output_name=k50_vae_ranking_220111\
  -k=10\
  --sample_size=5
