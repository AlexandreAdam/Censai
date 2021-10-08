import tensorflow as tf
import os, glob
import numpy as np
from censai.data.kappa_tng import decode_train as decode_kappa, decode_all as decode_kappa_info
from censai.data.alpha_tng_v2 import encode_examples
from censai import PhysicalModel
from datetime import datetime
import json


def main(args):
    kappa_datasets = []
    for path in args.kappa_datasets:
        files = glob.glob(os.path.join(path, "*.tfrecords"))
        np.random.shuffle(files)
        # Read concurrently from multiple records
        files = tf.data.Dataset.from_tensor_slices(files).shuffle(len(files), reshuffle_each_iteration=True)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type), block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)
        kappa_datasets.append(dataset.shuffle(args.buffer_size, reshuffle_each_iteration=True))
    kappa_dataset = tf.data.experimental.sample_from_datasets(kappa_datasets, weights=args.kappa_datasets_weights)
    # Read off global parameters from first example in dataset
    for example in kappa_dataset.map(decode_kappa_info):
        kappa_fov = example["kappa fov"]
        kappa_pixels = example["kappa pixels"]
        break
    kappa_dataset = kappa_dataset.map(decode_kappa).batch(args.batch_size)

    phys = PhysicalModel(image_fov=kappa_fov, pixels=kappa_pixels, kappa_fov=kappa_fov, method="conv2d")
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    for batch, kappa in enumerate(kappa_dataset):
        alpha = tf.concat(phys.deflection_angle(kappa), axis=-1)

        records = encode_examples(
            kappa=kappa,
            alpha=alpha,
            kappa_fov=kappa_fov
        )
    shard = batch * args.batch_size // args.shard_size
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{shard:01d}.tfrecords"), options=options) as writer:
        for record in records:
            writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--batch_size",                 default=1,          type=int,    help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--kappa_datasets",             required=True,      nargs="+",              help="Path to kappa tfrecords directories")
    parser.add_argument("--kappa_datasets_weights",     default=None,       nargs="+", type=float,  help="How much to sample from a dataset vs another. Must sum to 1/")
    parser.add_argument("--output_dir",                  required=True,     help="Path where tfrecords are stored")
    parser.add_argument("--compression_type",           default="GZIP",     help="Default is no compression. Use 'GZIP' to compress data")
    parser.add_argument("--shard_size",                 default=1000,       type=int,   help="Number of example stored in a shard")

    # Physics params
    parser.add_argument("--z_source",       default=2.379,  type=float)
    parser.add_argument("--z_lens",         default=0.4457, type=float)

    # Reproducibility params
    parser.add_argument("--seed",           default=None,   type=int,       help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override",  default=None,                   help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        with open(args.json_override, "r") as f:
            json_override = json.load(f)
        args_dict = vars(args)
        args_dict.update(json_override)
    with open(os.path.join(args.output_dir, "script_params.json"), "w") as f:
        args_dict = vars(args)
        json.dump(args_dict, f)

    main(args)
