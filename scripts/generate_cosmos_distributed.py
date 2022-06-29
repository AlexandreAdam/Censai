import tensorflow as tf
import json
from censai.models import VAE
from censai.utils import _bytes_feature, _int64_feature
import os
import time
from datetime import datetime

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def main(args):
    if THIS_WORKER > 1:
        time.sleep(5)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if args.seed is not None:
        tf.random.set_seed(args.seed)
    # Load first stage and freeze weights
    with open(os.path.join(args.cosmos_first_stage_vae, "model_hparams.json"), "r") as f:
        cosmos_vae_hparams = json.load(f)
    cosmos_vae = VAE(**cosmos_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=cosmos_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.cosmos_first_stage_vae, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
    cosmos_vae.trainable = False

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for _ in range((THIS_WORKER - 1) * args.batch_size, args.len_dataset, N_WORKERS * args.batch_size):
            galaxies = tf.nn.relu(cosmos_vae.sample(args.batch_size))
            galaxies /= tf.reduce_max(galaxies, axis=(1, 2, 3), keepdims=True)
            for j in range(args.batch_size):
                features = {
                    "image": _bytes_feature(galaxies[j].numpy().tobytes()),
                    "height": _int64_feature(galaxies.shape[1]),
                }
                record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                writer.write(record)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--cosmos_first_stage_vae",      required=True)
    parser.add_argument("--output_dir",                 required=True)
    parser.add_argument("--len_dataset",                required=True,  type=int)
    parser.add_argument("--batch_size",                 default=10,     type=int)
    parser.add_argument("--compression_type",           default="GZIP")

    parser.add_argument("--seed",           default=None,       type=int,   help="Random seed for numpy and tensorflow")

    args = parser.parse_args()

    main(args)