import tensorflow as tf
import json
from censai.models import VAE
from censai.data.kappa_tng import encode_examples
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

    # Load first stage
    with open(os.path.join(args.kappa_first_stage_vae, "model_hparams.json"), "r") as f:
        kappa_vae_hparams = json.load(f)
    kappa_vae = VAE(**kappa_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, args.kappa_first_stage_vae, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()
    kappa_vae.trainable = False

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for _ in range((THIS_WORKER - 1) * args.batch_size, args.len_dataset, N_WORKERS * args.batch_size):
            kappa = 10 ** kappa_vae.sample(args.batch_size)
            # Most important info to records are kappa and kappa fov, the rest are just fill-ins
            # to match same description as previous TNG tfrecords
            records = encode_examples(
                kappa=kappa,
                einstein_radius_init=[0.]*args.batch_size,
                einstein_radius=[0.]*args.batch_size,
                rescalings=[0.]*args.batch_size,
                z_source=0.,
                z_lens=0.,
                kappa_fov=args.kappa_fov,
                sigma_crit=0.,
                kappa_ids=[0]*args.batch_size
            )
            for record in records:
                writer.write(record)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--kappa_first_stage_vae",      required=True)
    parser.add_argument("--output_dir",                 required=True)
    parser.add_argument("--len_dataset",                required=True,  type=int)
    parser.add_argument("--batch_size",                 default=10,     type=int)
    parser.add_argument("--compression_type",           default="GZIP")

    # Physical params, should match training set
    parser.add_argument("--kappa_fov",      default=18,         type=float)

    parser.add_argument("--seed",           default=None,       type=int,   help="Random seed for numpy and tensorflow")

    args = parser.parse_args()
    main(args)