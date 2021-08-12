from censai.models import VAE, VAESecondStage
from censai.data.cosmos import encode_cosmos
import numpy as np
import tensorflow as tf
import os, glob
import json


def main(args):
    with open(os.path.join(args.first_stage_model_id, "model_hparams.json"), "r") as f:
        vae_hparams = json.load(f)
    # load weights
    vae = VAE(**vae_hparams)
    ckpt1 = tf.train.Checkpoint(net=vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.first_stage_model_id, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
    model1_name = os.path.split(args.first_stage_model_id)[-1]

    if args.second_stage_model_id is not None:
        with open(os.path.join(args.second_stage_model_id, "model_hparams.json"), "r") as f:
            vae2_hparams = json.load(f)
        vae2 = VAESecondStage(**vae2_hparams)
        ckpt2 = tf.train.Checkpoint(net=vae2)
        checkpoint_manager2 = tf.train.CheckpointManager(ckpt2, args.first_stage_model_id, 1)
        checkpoint_manager2.checkpoint.restore(checkpoint_manager2.latest_checkpoint).expect_partial()
        model2_name = os.path.split(args.second_stage_model_id)[-1]


    y_pred = vae.sample(args.sampling_size)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--first_stage_model_id",           required=True,                          help="Path to first stage model checkpoint directory.")
    parser.add_argument("--second_stage_model_id",          default=None,                           help="Path to second stage VAE. Optional.")
    parser.add_argument("--batch_size",                     default=20,         type=int,           help="Number of samples to generate at a given time.")
    parser.add_argument("--n_records",                      default=81,         type=int,           help="Number of individual record file to create.")
    parser.add_argument("--total_items",                    required=True,      type=int,           help="Total number of items to generate")
    parser.add_argument("--compression_type",               default=None,                           help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--seed",                           default=None,       type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)
