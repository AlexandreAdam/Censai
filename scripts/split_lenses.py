import tensorflow as tf
from censai.data.lenses_tng_v2 import encode_examples, decode_all
import os, glob
import math


def main(args):
    files = [glob.glob(os.path.join(args.dataset, "*.tfrecords"))]
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files).shuffle(len(files), reshuffle_each_iteration=False)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    total_items = 0
    for _ in dataset:
        total_items += 1

    train_items = math.floor(args.train_split * total_items)

    dataset = dataset.shuffle(args.buffer_size, reshuffle_each_iteration=False)
    train_dataset = dataset.take(train_items).batch(1)
    val_dataset = dataset.skip(train_items).batch(1)

    train_dir = args.dataset + "_train"
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    val_dir = args.dataset + "_val"
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    train_shards = train_items // args.examples_per_shard + 1 * (train_items % args.examples_per_shard > 0)
    for shard in range(train_shards):
        data = train_dataset.skip(shard * args.examples_per_shard).take(args.examples_per_shard)
        with tf.io.TFRecordWriter(os.path.join(train_dir, f"data_{shard:20d}.tfrecords"), options=options) as writer:
            for example in data.map(decode_all):
                record = encode_examples(
                    kappa=example["kappa"],
                    galaxies=example["source"],
                    lensed_images=example["lens"],
                    z_source=example["z source"],
                    z_lens=example["z lens"],
                    image_fov=example["image fov"],
                    kappa_fov=example["kappa fov"],
                    source_fov=example["source fov"],
                    noise_rms=example["noise rms"],
                    psf_sigma=example["psf sigma"])[0]
                writer.write(record)
    val_shards = (total_items - train_items) // args.examples_per_shard + 1 * ((total_items - train_items) % args.examples_per_shard > 0)
    for shard in range(val_shards):
        data = val_dataset.skip(shard * args.examples_per_shard).take(args.examples_per_shard)
        with tf.io.TFRecordWriter(os.path.join(val_dir, f"data_{shard:20d}.tfrecords"), options=options) as writer:
            for example in data.map(decode_all):
                record = encode_examples(
                    kappa=example["kappa"],
                    galaxies=example["source"],
                    lensed_images=example["lens"],
                    z_source=example["z source"],
                    z_lens=example["z lens"],
                    image_fov=example["image fov"],
                    kappa_fov=example["kappa fov"],
                    source_fov=example["source fov"],
                    noise_rms=example["noise rms"],
                    psf_sigma=example["psf sigma"])[0]
                writer.write(record)
    with open(os.path.join(train_dir, "dataset_size.txt"), "w") as f:
        f.write(f"{train_items:d}")
    with open(os.path.join(val_dir, "dataset_size.txt"), "w") as f:
        f.write(f"{total_items-train_items:d}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--train_split", default=0.9, type=float, help="Fraction of the dataset in the training set")
    parser.add_argument("--buffer_size", default=10000, type=int)
    parser.add_argument("--examples_per_shard", default=10000,  type=int,       help="Number of example to store in a single shard")

    args = parser.parse_args()

    main(args)
