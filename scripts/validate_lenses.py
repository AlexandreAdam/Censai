import tensorflow as tf
from censai.data.lenses_tng_v2 import decode_all, decode_physical_model_info
from censai.utils import _bytes_feature, _int64_feature, _float_feature
import os, glob, time
from datetime import datetime
from censai.definitions import DTYPE

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    if THIS_WORKER > 1:
        time.sleep(5)
    output_dir = args.dataset + "_validated"
    if THIS_WORKER == 1 and os.path.exists(os.path.join(output_dir, "shard_size.txt")):
        os.remove(os.path.join(output_dir, "shard_size.txt"))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    files = glob.glob(os.path.join(args.dataset, "*.tfrecords"))
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)
    for example in dataset.map(decode_physical_model_info):
        lens_pixels = example["pixels"].numpy()
        break
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    kept = 0
    current_dataset = dataset.skip((THIS_WORKER-1) * args.example_per_worker).take(args.example_per_worker)

    # setup mask for edge detection
    x = tf.range(lens_pixels, dtype=DTYPE) - lens_pixels//2 + 0.5 * lens_pixels % 2
    x, y = tf.meshgrid(x, x)
    edge = lens_pixels//2 - args.edge
    mask = (x > edge) | (x < -edge) | (y > edge) | (y < -edge)
    mask = tf.cast(mask[..., None], DTYPE)
    with tf.io.TFRecordWriter(os.path.join(output_dir, f"data_{THIS_WORKER:02d}.tfrecords"), options) as writer:
        for example in current_dataset.map(decode_all):
            im_area = tf.reduce_sum(tf.cast(example["lens"] > args.signal_threshold, tf.float32)) * (example["image fov"] / tf.cast(example["pixels"], DTYPE)) ** 2
            src_area = tf.reduce_sum(tf.cast(example["source"] > args.signal_threshold, tf.float32)) * (example["source fov"] / tf.cast(example["src pixels"], DTYPE)) ** 2
            magnification = im_area / src_area
            if magnification < args.min_magnification:
                continue
            if tf.reduce_max(example["lens"] * mask) > args.edge_signal_tolerance:
                continue
            kept += 1
            features = {
                "kappa": _bytes_feature(example["kappa"].numpy().tobytes()),
                "source": _bytes_feature(example["source"].numpy().tobytes()),
                "lens": _bytes_feature(example["lens"].numpy().tobytes()),
                "z source": _float_feature(example["z source"].numpy()),
                "z lens": _float_feature(example["z lens"].numpy()),
                "image fov": _float_feature(example["image fov"].numpy()),  # arc seconds
                "kappa fov": _float_feature(example["kappa fov"].numpy()),  # arc seconds
                "source fov": _float_feature(example["source fov"].numpy()),  # arc seconds
                "src pixels": _int64_feature(example["source"].shape[0]),
                "kappa pixels": _int64_feature(example["kappa"].shape[0]),
                "pixels": _int64_feature(example["lens"].shape[0]),
                "noise rms": _float_feature(example["noise rms"].numpy()),
                "psf sigma": _float_feature(example["psf sigma"].numpy()),
            }
            serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
            record = serialized_output.SerializeToString()
            writer.write(record)
    print(f"Finished worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}, kept {kept:d} examples")

    with open(os.path.join(output_dir, "shard_size.txt"), "a") as f:
        f.write(f"{THIS_WORKER} {kept:d}\n")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset",                required=True,                  help="Path to dataset")
    parser.add_argument("--min_magnification",      default=3,      type=float,     help="Minimum magnification for the lens")
    parser.add_argument("--signal_threshold",       default=0.1,    type=float,     help="Threshold for signal to be considered as such")
    parser.add_argument("--example_per_worker",     default=100000, type=int,       help="")
    parser.add_argument("--compression_type",       default="GZIP")
    parser.add_argument("--block_length",           default=1,      type=int)
    parser.add_argument("--border_pixels",          default=5,      type=int,       help="Check that intensity fall off at border")
    parser.add_argument("--edge",                   default=5,      type=int,       help="Number of pixels considered as egde pixels")
    parser.add_argument("--edge_signal_tolerance",  default=0.2,    type=float,     help="If maximum along the edge is above this value, example is rejected.")

    args = parser.parse_args()

    distributed_strategy(args)
