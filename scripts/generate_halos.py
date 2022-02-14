from censai.models import VAE
from censai.utils import _int64_feature, _bytes_feature
import numpy as np
import tensorflow as tf
import os
import json


def main(args):
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with open(os.path.join(args.first_stage_model_id, "model_hparams.json"), "r") as f:
        vae_hparams = json.load(f)
    # load weights
    vae = VAE(**vae_hparams)
    ckpt1 = tf.train.Checkpoint(net=vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.first_stage_model_id, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

    n_batch = args.total_items // args.batch_size
    batch_per_record = args.n_records // n_batch
    last_record_n_batch = batch_per_record + n_batch % args.n_records

    for record in range(args.n_records - 1):
        with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{record:02d}.tfrecords"), options) as writer:
            for batch in range(batch_per_record):
                z = tf.random.normal(shape=[args.batch_size, vae.latent_size])
                kappa_batch = vae.decode(z)
                for kappa in kappa_batch:
                    features = {
                        "kappa": _bytes_feature(kappa.numpy().tobytes()),
                        "kappa pixels": _int64_feature(kappa.shape[0]),
                    }

                    record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                    writer.write(record)

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{args.n_record-1:02d}.tfrecords"), options) as writer:
        for batch in range(last_record_n_batch):
            z = tf.random.normal(shape=[args.batch_size, vae.latent_size])
            kappa_batch = vae.decode(z)
            for kappa in kappa_batch:
                features = {
                    "kappa": _bytes_feature(kappa.numpy().tobytes()),
                    "kappa pixels": _int64_feature(kappa.shape[0]),
                }

                record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                writer.write(record)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",                     required=True)
    parser.add_argument("--first_stage_model_id",           required=True,                          help="Path to first stage model checkpoint directory.")
    parser.add_argument("--batch_size",                     default=20,         type=int,           help="Number of samples to generate at a given time.")
    parser.add_argument("--n_records",                      default=81,         type=int,           help="Number of individual record file to create.")
    parser.add_argument("--total_items",                    required=True,      type=int,           help="Total number of items to generate")
    parser.add_argument("--compression_type",               default=None,                           help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--seed",                           default=None,       type=int)

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)
