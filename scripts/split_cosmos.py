# used mainly for cosmos, since we can more easily split kappa maps
import tensorflow as tf
import math
import glob, os


def main(args):
    files = [glob.glob(os.path.join(args.dataset, "*.tfrecords"))]
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    total_items = 0
    for _ in dataset:
        total_items += 1

    train_items = math.floor(args.train_split * total_items)

    train_dataset = dataset.take(train_items).shuffle(args.buffer_size)
    test_dataset = dataset.skip(train_items)

    train_dir = args.dataset + "_train"
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    test_dir = args.dataset + "_test"
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(train_dir, "data.tfrecords"), options=options) as writer:
        for record in train_dataset:
            writer.write(record)
    with tf.io.TFRecordWriter(os.path.join(test_dir, "data.tfrecords"), options=options) as writer:
        for record in test_dataset:
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
    parser.add_argument("--buffer_size", default=1000, type=int)

    args = parser.parse_args()
    main(args)