import tensorflow as tf
import os, glob
import numpy as np
from censai import PhysicalModelv2
from censai.data.cosmos import preprocess, decode
from censai.data import NISGenerator
from censai.data.lenses_tng_v3 import encode_examples
from scipy.signal.windows import tukey
from scipy.stats import truncnorm
from datetime import datetime
import json, time


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

    min_theta_e = 0.1 * args.image_fov if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.45 * args.image_fov if args.max_theta_e is None else args.max_theta_e

    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    cosmos = tf.data.TFRecordDataset(cosmos_files)
    n_galaxies = 0
    for _ in cosmos:  # count the number of samples in the dataset
        n_galaxies += 1
    cosmos = cosmos.map(decode).map(preprocess)
    if args.shuffle_cosmos:
        cosmos = cosmos.shuffle(buffer_size=args.buffer_size, reshuffle_each_iteration=True)
    cosmos = cosmos.batch_size(args.batch_size)

    window = tukey(args.src_pixels, alpha=args.tukey_alpha)
    window = np.outer(window, window)
    phys = PhysicalModelv2(
        image_fov=args.image_fov,
        kappa_fov=args.kappa_fov,
        src_fov=args.source_fov,
        pixels=args.lens_pixels,
        kappa_pixels=args.kappa_pixels,
        src_pixels=args.src_pixels,
        method="conv2d"
    )
    noise_a = (args.noise_rms_min - args.noise_rms_mean) / args.noise_rms_std
    noise_b = (args.noise_rms_max - args.noise_rms_mean) / args.noise_rms_std
    psf_a = (args.psf_fwhm_min - args.psf_fwhm_mean) / args.psf_fwhm_std
    psf_b = (args.psf_fwhm_max - args.psf_fwhm_mean) / args.psf_fwhm_std

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch_size, args.len_dataset, N_WORKERS * args.batch_size):
            batch_index = np.random.randint(0, n_galaxies//args.batch_size)
            for galaxies, psf, ps in cosmos.skip(batch_index):  # only way to take the first batch is to fake a for loop
                break
            galaxies = window[np.newaxis, ..., np.newaxis] * galaxies

            noise_rms = truncnorm.rvs(noise_a, noise_b, loc=args.noise_rms_mean, scale=args.noise_rms_std, size=args.batch_size)
            sigma = truncnorm.rvs(psf_a, psf_b, loc=args.psf_fwhm_mean, scale=args.psf_fwhm_std, size=args.batch_size)
            psf = phys.psf_models(sigma, cutout_size=args.psf_cutout_size)

            batch_size = galaxies.shape[0]
            _r = tf.random.uniform(shape=[batch_size, 1, 1], minval=0, maxval=args.max_shift)
            _theta = tf.random.uniform(shape=[batch_size, 1, 1], minval=-np.pi, maxval=np.pi)
            x0 = _r * tf.math.cos(_theta)
            y0 = _r * tf.math.sin(_theta)
            ellipticity = tf.random.uniform(shape=[batch_size, 1, 1], minval=0., maxval=args.max_ellipticity)
            phi = tf.random.uniform(shape=[batch_size, 1, 1], minval=-np.pi, maxval=np.pi)
            einstein_radius = tf.random.uniform(shape=[batch_size, 1, 1], minval=min_theta_e, maxval=max_theta_e)

            kappa = kappa_gen.kappa_field(x0, y0, ellipticity, phi, einstein_radius)

            lensed_images = phys.noisy_forward(galaxies, kappa, noise_rms=noise_rms, psf=psf)

            records = encode_examples(
                kappa=kappa,
                galaxies=galaxies,
                lensed_images=lensed_images,
                z_source=args.z_source,
                z_lens=args.z_lens,
                image_fov=phys.image_fov,
                kappa_fov=phys.kappa_fov,
                source_fov=args.source_fov,
                noise_rms=noise_rms,
                psf=psf
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",         required=True,      type=str,   help="Path to output directory")
    parser.add_argument("--len_dataset",        required=True,      type=int,   help="Size of the dataset")
    parser.add_argument("--batch_size",         default=1,          type=int,   help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--cosmos_dir",         required=True,      type=str,   help="Path to directory of galaxy brightness distribution tfrecords "
                                                                                     "(output of cosmos_to_tfrecors.py)")
    parser.add_argument("--compression_type",   default=None,                   help="Default is no compression. Use 'GZIP' to compress data")

    # Physical model params
    parser.add_argument("--lens_pixels",        default=512,    type=int,       help="Number of pixels on a side of the kappa map.")
    parser.add_argument("--kappa_pixels",       default=128,    type=int,       help="Size of the lens postage stamp.")
    parser.add_argument("--src_pixels",         default=128,    type=int,       help="Size of Cosmos postage stamps")
    parser.add_argument("--kappa_fov",          default=17.42,  type=float,     help="Field of view of kappa map in arcseconds")
    parser.add_argument("--image_fov",          default=17.42,  type=float,     help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--source_fov",         default=6,      type=float,     help="Field of view of the source plane in arc seconds")
    parser.add_argument("--noise_rms_min",      default=0.001,  type=float,     help="Minimum white noise RMS added to lensed image")
    parser.add_argument("--noise_rms_max",      default=0.1,    type=float,     help="Maximum white noise RMS added to lensed image")
    parser.add_argument("--noise_rms_mean",     default=0.01,   type=float,     help="Maximum white noise RMS added to lensed image")
    parser.add_argument("--noise_rms_std",      default=0.03,   type=float,     help="Maximum white noise RMS added to lensed image")
    parser.add_argument("--psf_cutout_size",    default=32,     type=int,       help="Size of the cutout for the PSF (arceconds)")
    parser.add_argument("--psf_fwhm_min",       default=0.04,   type=float,     help="Minimum std of gaussian psf (arceconds)")
    parser.add_argument("--psf_fwhm_max",       default=0.3,    type=float,     help="Minimum std of gaussian psf (arceconds)")
    parser.add_argument("--psf_fwhm_mean",      default=0.08,   type=float,     help="Mean for the distribution of std of the gaussian psf (arceconds)")
    parser.add_argument("--psf_fwhm_std",       default=0.08,   type=float,     help="Std for distribution of stf of the gaussian psf (arceconds)")

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
    parser.add_argument("--z_source",           default=1.5,       type=float)
    parser.add_argument("--z_lens",             default=0.5,       type=float)

    # Reproducibility params
    parser.add_argument("--seed",               default=None,       type=int,   help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override",      default=None,       nargs="+",  help="A json filepath that will override every command line parameters. "
                                                                                     "Useful for reproducibility")

    args = parser.parse_args()
    if THIS_WORKER > 1:
        time.sleep(5)
    if not os.path.isdir(args.output_dir):
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
