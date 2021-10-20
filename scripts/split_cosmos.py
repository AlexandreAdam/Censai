# used mainly for cosmos, since we can more easily split kappa maps
import tensorflow as tf
import math
import glob, os
from censai.data.cosmos import decode_image
from censai.utils import _bytes_feature, _int64_feature


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
    train_dataset = dataset.take(train_items)
    test_dataset = dataset.skip(train_items)

    train_dir = args.dataset + "_train"
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    test_dir = args.dataset + "_test"
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    train_shards = train_items // args.examples_per_shard + 1 * (train_items % args.examples_per_shard > 0)
    for shard in range(train_shards):
        data = train_dataset.skip(shard * args.examples_per_shard).take(args.examples_per_shard)
        with tf.io.TFRecordWriter(os.path.join(train_dir, f"data_{shard:20d}.tfrecords"), options=options) as writer:
            for image in data.map(decode_image):
                features = {
                    "image": _bytes_feature(image.numpy().tobytes()),
                    "height": _int64_feature(image.shape[0]),
                }
                record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                writer.write(record)
    test_shards = (total_items - train_items) // args.examples_per_shard + 1 * ((total_items - train_items) % args.examples_per_shard > 0)
    for shard in range(test_shards):
        data = test_dataset.skip(shard * args.examples_per_shard).take(args.examples_per_shard)
        with tf.io.TFRecordWriter(os.path.join(test_dir, f"data_{shard:20d}.tfrecords"), options=options) as writer:
            for example in data.map(decode_image):
                features = {
                    "image": _bytes_feature(image.numpy().tobytes()),
                    "height": _int64_feature(image.shape[0]),
                }
                record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                writer.write(record)

    with open(os.path.join(train_dir, "dataset_size.txt"), "w") as f:
        f.write(f"{train_items:d}")

    with open(os.path.join(test_dir, "dataset_size.txt"), "w") as f:
        f.write(f"{total_items-train_items:d}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--compression_type",   default=None)
    parser.add_argument("--train_split", default=0.9, type=float, help="Fraction of the dataset in the training set")
    parser.add_argument("--buffer_size", default=50000, type=int)
    parser.add_argument("--examples_per_shard",   default=10000,  type=int,       help="Number of example to store in a single shard")

    args = parser.parse_args()
    main(args)