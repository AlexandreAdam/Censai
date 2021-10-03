import tensorflow as tf
import os
import numpy as np
from censai.data import NISGenerator
from censai.data.alpha_tng import encode_examples
from censai import PhysicalModel
from datetime import datetime

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    kappa_gen = NISGenerator(
        kappa_fov=args.kappa_fov,
        pixels=args.pixels,
        z_source=args.z_source,
        z_lens=args.z_lens
    )

    min_theta_e = 0.1 * args.image_fov if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.45 * args.image_fov if args.max_theta_e is None else args.max_theta_e

    phys = PhysicalModel(image_fov=args.kappa_fov, pixels=args.pixels,
                         kappa_fov=args.kappa_fov, method="conv2d")

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"kappa_alpha_{THIS_WORKER}.tfrecords"), options=options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch, args.len_dataset, N_WORKERS * args.batch):
            _r = tf.random.uniform(shape=[args.batch, 1, 1], minval=0, maxval=args.max_shift)
            _theta = tf.random.uniform(shape=[args.batch, 1, 1], minval=-np.pi, maxval=np.pi)
            x0 = _r * tf.math.cos(_theta)
            y0 = _r * tf.math.sin(_theta)
            ellipticity = tf.random.uniform(shape=[args.batch, 1, 1], minval=0., maxval=args.max_ellipticity)
            phi = tf.random.uniform(shape=[args.batch, 1, 1], minval=-np.pi, maxval=np.pi)
            einstein_radius = tf.random.uniform(shape=[args.batch, 1, 1], minval=min_theta_e, maxval=max_theta_e)

            kappa = kappa_gen.kappa_field(x0, y0, ellipticity, phi, einstein_radius)

            alpha = tf.concat(phys.deflection_angle(kappa)[2:], axis=-1)

            records = encode_examples(
                kappa=kappa,
                alpha=alpha,
                rescalings=[1] * args.batch,
                kappa_ids=[-1] * args.batch,
                einstein_radius=einstein_radius.numpy().squeeze(),
                image_fov=args.image_fov,
                kappa_fov=args.kappa_fov
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",         required=True,                  help="Path where tfrecords are stored")
    parser.add_argument("--compression_type",   default=None,                   help="Default is no compression. Use 'GZIP' to compress data")
    parser.add_argument("--len_dataset",        required=True,  type=int,       help="Size of the dataset")

    # Physical Model params
    parser.add_argument("--image_fov",          default=20,     type=float,     help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--kappa_fov",          default=22.2,   type=float,     help="Field of view of kappa map in arcseconds")

    # Data generation params
    parser.add_argument("--pixels",             default=512,    type=int,       help="Size of the tensors.")
    parser.add_argument("--batch",              default=1,      type=int,       help="Number of label maps to be computed at the same time")
    parser.add_argument("--max_shift",          default=1.5,    type=float,       help="Max shift for the center of the kappa map.")
    parser.add_argument("--max_ellipticity",    default=0.6,    type=float,     help="Maximum ellipticty of density profile.")
    parser.add_argument("--max_theta_e",        default=None,   type=float,     help="Maximum allowed Einstein radius, default is 35 percent of image fov")
    parser.add_argument("--min_theta_e",        default=None,   type=float,     help="Minimum allowed Einstein radius, default is 5 percent of image fov")

    # Physics params
    parser.add_argument("--z_source",           default=2.379,  type=float)
    parser.add_argument("--z_lens",             default=0.4457, type=float)

    # Reproducibility params
    parser.add_argument("--seed",               default=None,   type=int,       help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override",      default=None,                   help="A json filepath that will override every command line parameters. Useful for reproducibility")

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir) and THIS_WORKER <= 1:
        os.mkdir(args.output_dir)
    if args.seed is not None:
        tf.random.set_seed(args.seed)
        np.random.seed(args.seed)
    if args.json_override is not None:
        import json
        with open(args.json_override, "r") as f:
            json_override = json.load(f)
        args_dict = vars(args)
        args_dict.update(json_override)
    if THIS_WORKER == 1:
        import json
        with open(os.path.join(args.output_dir, "script_params.json"), "w") as f:
            args_dict = vars(args)
            json.dump(args_dict, f)

    distributed_strategy(args)
