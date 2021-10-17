import tensorflow as tf
import os, glob
import numpy as np
from censai import PhysicalModel
from censai.data.cosmos import preprocess_image as preprocess_cosmos, decode_image as decode_cosmos, decode_shape as decode_cosmos_info
from censai.data.lenses_tng_v2 import encode_examples
from censai.data.kappa_tng import decode_train as decode_kappa, decode_all as decode_kappa_info
from scipy.signal.windows import tukey
from censai.definitions import DTYPE
from datetime import datetime
import json


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
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
        kappa_fov = example["kappa fov"].numpy()
        kappa_pixels = example["kappa pixels"].numpy()
        break
    kappa_dataset = kappa_dataset.map(decode_kappa).batch(args.batch_size)

    cosmos_datasets = []
    for path in args.cosmos_datasets:
        files = glob.glob(os.path.join(path, "*.tfrecords"))
        np.random.shuffle(files)
        # Read concurrently from multiple records
        files = tf.data.Dataset.from_tensor_slices(files).shuffle(len(files), reshuffle_each_iteration=True)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type), block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)
        cosmos_datasets.append(dataset.shuffle(args.buffer_size, reshuffle_each_iteration=True))
    cosmos_dataset = tf.data.experimental.sample_from_datasets(cosmos_datasets, weights=args.cosmos_datasets_weights)
    # Read off global parameters from first example in dataset
    for src_pixels in cosmos_dataset.map(decode_cosmos_info):
        src_pixels = src_pixels.numpy()
        break
    cosmos_dataset = cosmos_dataset.map(decode_cosmos).map(preprocess_cosmos).batch(args.batch_size)

    window = tukey(src_pixels, alpha=args.tukey_alpha)
    window = np.outer(window, window)[np.newaxis, ..., np.newaxis]
    window = tf.constant(window, dtype=DTYPE)

    phys = PhysicalModel(
        psf_sigma=args.psf_sigma,
        image_fov=kappa_fov,
        src_fov=args.source_fov,
        pixels=args.lens_pixels,
        kappa_pixels=kappa_pixels,
        src_pixels=src_pixels,
        kappa_fov=kappa_fov,
        method="conv2d"
    )

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch_size, args.len_dataset, N_WORKERS * args.batch_size):
            for galaxies in cosmos_dataset:  # select a random batch from our dataset that is reshuffled each iterations
                break
            for kappa in kappa_dataset:
                break
            galaxies = window * galaxies
            lensed_images = phys.noisy_forward(galaxies, kappa, noise_rms=args.noise_rms)

            records = encode_examples(
                kappa=kappa,
                galaxies=galaxies,
                lensed_images=lensed_images,
                z_source=args.z_source,
                z_lens=args.z_lens,
                image_fov=phys.image_fov,
                kappa_fov=phys.kappa_fov,
                source_fov=args.source_fov,
                noise_rms=args.noise_rms,
                psf_sigma=args.psf_sigma
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",                 required=True,                  type=str,   help="Path to output directory")
    parser.add_argument("--len_dataset",                required=True,                  type=int,   help="Size of the dataset")
    parser.add_argument("--batch_size",                 default=1,                     type=int,    help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--kappa_datasets",             required=True,      nargs="+",              help="Path to kappa tfrecords directories")
    parser.add_argument("--kappa_datasets_weights",     default=None,       nargs="+", type=float,  help="How much to sample from a dataset vs another. Must sum to 1/")
    parser.add_argument("--cosmos_datasets",            required=True,      nargs="+",              help="Path to galaxy tfrecords directories")
    parser.add_argument("--cosmos_datasets_weights",    default=None,       nargs="+", type=float,  help="How much to sample from a dataset vs another. Must sum to 1")
    parser.add_argument("--compression_type",           default="GZIP",                             help="Default is GZIP and should stay that way.")
    parser.add_argument("--block_length",               default=1,           type=int,              help="Number of example to read concurrently from a file")

    # Physical model params
    parser.add_argument("--lens_pixels",    default=512,        type=int,   help="Size of the lens postage stamp.")
    parser.add_argument("--source_fov",     default=6,          type=float, help="Field of view of the source plane in arc seconds")
    parser.add_argument("--noise_rms",      default=0.05,       type=float, help="White noise RMS added to lensed image")
    parser.add_argument("--psf_sigma",      default=0.06,       type=float, help="Sigma of psf in arcseconds")

    # Data generation params
    parser.add_argument("--buffer_size",    default=10000,       type=int,    help="buffer of shuffle, should be similar to number of examples per shard (at least greater than the largest shard)")
    parser.add_argument("--tukey_alpha",    default=0.,          type=float, help="Shape parameter of the Tukey window, representing the fraction of the "
                                                                                 "window inside the cosine tapered region. "
                                                                                 "If 0, the Tukey window is equivalent to a rectangular window. "
                                                                                 "If 1, the Tukey window is equivalent to a Hann window. "
                                                                                 "This window is used on cosmos postage stamps.")

    # Physics params
    parser.add_argument("--z_source",       default=2.379,      type=float)
    parser.add_argument("--z_lens",         default=0.4457,     type=float)

    # Reproducibility params
    parser.add_argument("--seed",           default=None,       type=int,   help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override",  default=None, nargs="+",        help="A json filepath that will override every command line parameters. "
                                                                                 "Useful for reproducibility")

    args = parser.parse_args()
    if not os.path.isdir(args.output_dir) and THIS_WORKER <= 1:
        os.mkdir(args.output_dir)
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        if isinstance(args.json_override, list):
            files = args.json_override
        else:
            files = [args.json_override,]
        for file in files:
            with open(file, "r") as f:
                json_override = json.load(f)
            args_dict = vars(args)
            args_dict.update(json_override)
    if THIS_WORKER == 1:
        import json
        with open(os.path.join(args.output_dir, "script_params.json"), "w") as f:
            args_dict = vars(args)
            json.dump(args_dict, f)

    distributed_strategy(args)
