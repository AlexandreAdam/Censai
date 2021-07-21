import tensorflow as tf
import os, glob
import numpy as np
from censai import PhysicalModel
from censai.data.cosmos import preprocess, decode
from censai.data import NISGenerator
from censai.data.lenses_tng import encode_examples
from scipy.signal.windows import tukey
from datetime import datetime
import json


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    kappa_gen = NISGenerator( # only used to generate pixelated kappa fields
        kappa_fov=args.kappa_fov,
        src_fov=args.source_fov,
        pixels=args.kappa_pixels,
        z_source=args.z_source,
        z_lens=args.z_lens
    )

    min_theta_e = 0.05 * args.image_fov if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.35 * args.image_fov if args.max_theta_e is None else args.max_theta_e

    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    n_galaxies = len(cosmos_files) * args.buffer_size
    cosmos = tf.data.TFRecordDataset(cosmos_files).map(decode).map(preprocess)
    if args.shuffle_cosmos:
        cosmos = cosmos.shuffle(buffer_size=args.buffer_size)
    cosmos = cosmos.batch(args.batch)

    window = tukey(args.src_pixels, alpha=args.tukey_alpha)
    window = np.outer(window, window)
    phys = PhysicalModel(
        psf_sigma=args.psf_sigma,
        image_fov=args.image_fov,
        kappa_fov=args.kappa_fov,
        src_fov=args.source_fov,
        pixels=args.lens_pixels,
        kappa_pixels=args.kappa_pixels,
        src_pixels=args.src_pixels,
        method="conv2d"
    )

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch, args.len_dataset, N_WORKERS * args.batch):
            batch_index = np.random.randint(0, n_galaxies//args.batch)
            for galaxies, psf, ps in cosmos.skip(batch_index):  # only way to take the first batch is to fake a for loop
                break
            galaxies = window[np.newaxis, ..., np.newaxis] * galaxies

            batch_size = galaxies.shape[0]
            _r = tf.random.uniform(shape=[batch_size, 1, 1], minval=0, maxval=args.max_shift)
            _theta = tf.random.uniform(shape=[batch_size, 1, 1], minval=-np.pi, maxval=np.pi)
            x0 = _r * tf.math.cos(_theta)
            y0 = _r * tf.math.sin(_theta)
            ellipticity = tf.random.uniform(shape=[batch_size, 1, 1], minval=0., maxval=args.max_ellipticity)
            phi = tf.random.uniform(shape=[batch_size, 1, 1], minval=-np.pi, maxval=np.pi)
            einstein_radius = tf.random.uniform(shape=[batch_size, 1, 1], minval=min_theta_e, maxval=max_theta_e)

            kappa = kappa_gen.kappa_field(x0, y0, ellipticity, phi, einstein_radius)

            lensed_images = phys.noisy_forward(galaxies, kappa, noise_rms=args.noise_rms)

            records = encode_examples(
                kappa=kappa,
                galaxies=galaxies,
                lensed_images=lensed_images,
                power_spectrum_cosmos=ps,
                einstein_radius_init=einstein_radius.numpy().squeeze(),
                einstein_radius=einstein_radius.numpy().squeeze(),
                rescalings=[1] * args.batch,
                z_source=args.z_source,
                z_lens=args.z_lens,
                image_fov=args.image_fov,
                kappa_fov=phys.kappa_fov,
                source_fov=args.source_fov,
                sigma_crit=kappa_gen.sigma_crit,  # 10^{10} M_sun / Mpc^2
                noise_rms=args.noise_rms,
                psf_sigma=args.psf_sigma,
                kappa_ids=[-1] * args.batch
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",         required=True,      type=str,   help="Path to output directory")
    parser.add_argument("--len_dataset",        required=True,      type=int,   help="Size of the dataset")
    parser.add_argument("--batch",              default=1,          type=int,   help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--cosmos_dir",         required=True,      type=str,   help="Path to directory of galaxy brightness distribution tfrecords "
                                                                                     "(output of cosmos_to_tfrecors.py)")
    parser.add_argument("--compression_type",   default=None,                   help="Default is no compression. Use 'GZIP' to compress data")

    # Physical model params
    parser.add_argument("--lens_pixels",        default=512,        type=int,   help="Number of pixels on a side of the kappa map.")
    parser.add_argument("--kappa_pixels",       default=128,        type=int,   help="Size of the lens postage stamp.")
    parser.add_argument("--src_pixels",         default=128,        type=int,   help="Size of Cosmos postage stamps")
    parser.add_argument("--kappa_fov",          default=22.2,       type=float, help="Field of view of kappa map in arcseconds")
    parser.add_argument("--image_fov",          default=20,         type=float, help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--source_fov",         default=3,          type=float,
                        help="Field of view of the source plane in arc seconds")
    parser.add_argument("--noise_rms",          default=0.3e-3,     type=float,
                        help="White noise RMS added to lensed image")
    parser.add_argument("--psf_sigma",          default=0.06,       type=float, help="Sigma of psf in arcseconds")

    # Data generation params
    parser.add_argument("--max_shift",          default=1.,         type=float, help="Maximum allowed shift of kappa map center in arcseconds")
    parser.add_argument("--max_ellipticity",    default=0.6,        type=float, help="Maximum ellipticty of density profile.")
    parser.add_argument("--max_theta_e",        default=None,       type=float, help="Maximum allowed Einstein radius, default is 35 percent of image fov")
    parser.add_argument("--min_theta_e",        default=None,       type=float, help="Minimum allowed Einstein radius, default is 5 percent of image fov")
    parser.add_argument("--shuffle_cosmos",     action="store_true",            help="Shuffle indices of cosmos dataset")
    parser.add_argument("--buffer_size",        default=1000,       type=int,   help="Should match example_per_shard when tfrecords were produced "
                                                                                     "(only used if shuffle_cosmos is called)")
    parser.add_argument("--tukey_alpha",        default=0.6,        type=float, help="Shape parameter of the Tukey window, representing the fraction of the "
                                                                                     "window inside the cosine tapered region. "
                                                                                     "If 0, the Tukey window is equivalent to a rectangular window. "
                                                                                     "If 1, the Tukey window is equivalent to a Hann window. "
                                                                                     "This window is used on cosmos postage stamps.")

    # Physics params
    parser.add_argument("--z_source",           default=2.379,      type=float)
    parser.add_argument("--z_lens",             default=0.4457,     type=float)

    # Reproducibility params
    parser.add_argument("--seed",               default=None,       type=int,   help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override",      default=None,       nargs="+",  help="A json filepath that will override every command line parameters. "
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
