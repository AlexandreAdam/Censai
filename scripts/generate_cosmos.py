from censai.models import VAE
from censai.data.cosmos import encode_cosmos
import numpy as np
import tensorflow as tf
import os, glob
import json


def main(args):
    files = []
    files.extend(glob.glob(os.path.join(args.dataset, "*.tfrecords")))
    np.random.shuffle(files)
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)

    # load weights
    vae = VAE(**vae_hparams)
    ckpt1 = tf.train.Checkpoint(net=vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, model, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

    model_name = os.path.split(model)[-1]


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
