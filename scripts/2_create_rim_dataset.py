import tensorflow as tf
import os, glob
import numpy as np
from censai import PhysicalModel
from censai.data.cosmos import preprocess, decode
from censai.data import AugmentedTNGKappaGenerator
from censai.data.lenses_tng import encode_examples
from scipy.signal.windows import tukey
from datetime import datetime


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    kappa_files = glob.glob(os.path.join(args.kappa_dir, "*.fits"))
    if os.path.exists(os.path.join(args.kappa_dir, "good_kappa.txt")):  # filter out bad data (see validate_kappa_maps script)
        good_kappa = np.loadtxt(os.path.join(args.kappa_dir, "good_kappa.txt"))
        kappa_ids = [int(os.path.split(kap)[-1].split("_")[1]) for kap in kappa_files]
        keep_kappa = [kap_id in good_kappa for kap_id in kappa_ids]
        kappa_files = [kap_file for i, kap_file in enumerate(kappa_files) if keep_kappa[i]]

    kappa_gen = AugmentedTNGKappaGenerator(
        kappa_fits_files=kappa_files,
        z_lens=args.z_lens,
        z_source=args.z_source,
        crop=args.crop,
        rotate_by=args.rotate_by,
        min_theta_e=args.min_theta_e,
        max_theta_e=args.max_theta_e,
        rescaling_size=args.rescaling_size,
        rescaling_theta_bins=args.bins
    )
    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    n_galaxies = len(cosmos_files) * args.buffer_size
    cosmos = tf.data.TFRecordDataset(cosmos_files).map(decode).map(preprocess)
    if args.shuffle_cosmos:
        cosmos = cosmos.shuffle(buffer_size=args.buffer_size)
    cosmos = cosmos.batch(args.batch)

    window = tukey(args.src_pixels, alpha=args.tukey_alpha)
    window = np.outer(window, window)
    phys = PhysicalModel(image_fov=args.image_fov, src_side=args.source_fov, pixels=kappa_gen.crop_pixels,
                         src_pixels=args.src_pixels, kappa_fov=kappa_gen.kappa_fov, method="conv2d")

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{this_worker}.tfrecords")) as writer:
        print(f"Started worker {this_worker} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((this_worker-1) * args.batch, args.len_dataset, N_WORKERS * args.batch):
            batch_index = np.random.randint(0, n_galaxies//args.batch)
            for galaxies, psf, ps in cosmos.skip(batch_index):  # only way to take the first batch is to fake a for loop
                break
            galaxies = window[np.newaxis, ..., np.newaxis] * galaxies

            batch_size = galaxies.shape[0]  # Batch size will differ if we selected last batch of galaxy dataset
            kappa, einstein_radius, rescaling_factors, kappa_ids, einstein_radius_init = kappa_gen.draw_batch(
                batch_size, rescale=True, shift=True, rotate=args.rotate, random_draw=True, return_einstein_radius_init=True)
            lensed_images = phys.noisy_forward(galaxies, kappa, noise_rms=args.noise_rms)

            records = encode_examples(
                kappa=kappa,
                galaxies=galaxies,
                lensed_images=lensed_images,
                power_spectrum_cosmos=ps,
                einstein_radius_init=einstein_radius_init,
                einstein_radius=einstein_radius,
                rescalings=rescaling_factors,
                z_source=args.z_source,
                z_lens=args.z_lens,
                image_fov=args.image_fov,
                kappa_fov=phys.kappa_fov,
                sigma_crit=kappa_gen.sigma_crit,
                noise_rms=args.noise_rms,
                kappa_ids=kappa_ids
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",  required=True, type=str, help="Path to output directory")
    parser.add_argument("--len_dataset", required=True, type=int, help="Size of the dataset")
    parser.add_argument("--kappa_dir",   required=True, type=str, help="Path to directory of kappa fits files")
    parser.add_argument("--cosmos_dir",  required=True, type=str,
                        help="Path to directory of galaxy brightness distribution tfrecords "
                             "(output of cosmos_to_tfrecors.py)")
    # Physical model params
    parser.add_argument("--src_pixels",  default=128,    type=int, help="Size of Cosmos postage stamps")
    parser.add_argument("--image_fov",   default=20,     type=float,
                        help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--source_fov",  default=3,      type=float,
                        help="Field of view of the source plane in arc seconds")
    parser.add_argument("--noise_rms",   default=0.3e-3, type=float,
                        help="White noise RMS added to lensed image")
    #TODO add an option to change color of the noise to match COSMOS color

    # Data generation params
    parser.add_argument("--crop",           default=0,      type=int,
                        help="Crop kappa map by 2*N pixels. After crop, the size of the kappa map "
                             "should correspond to pixel argument "
                             "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--rotate",         action="store_true", help="Rotate the kappa map")
    parser.add_argument("--rotate_by",      default="90",
                        help="'90': will rotate by a multiple of 90 degrees. 'uniform' will rotate by any angle, "
                             "with nearest neighbor interpolation and zero padding")
    parser.add_argument("--shuffle_cosmos", action="store_true", help="Shuffle indices of cosmos dataset")
    parser.add_argument("--buffer_size",    default=1000,   type=int,
                        help="Should match example_per_shard when tfrecords were produced "
                             "(only used if shuffle_cosmos is called)")
    parser.add_argument("--batch",          default=1,      type=int,
                        help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--tukey_alpha",    default=0.6,    type=float,  # help from scipy own documentation
                        help="Shape parameter of the Tukey window, representing the fraction of the "
                             "window inside the cosine tapered region. "
                             "If 0, the Tukey window is equivalent to a rectangular window. "
                             "If 1, the Tukey window is equivalent to a Hann window. "
                             "This window is used on cosmos postage stamps.")
    parser.add_argument("--bins",           default=10,     type=int,
                        help="Number of bins to estimate Einstein radius distribution of a kappa given "
                             "a set of rescaling factors.")
    parser.add_argument("--rescaling_size", default=100,    type=int,
                        help="Number of rescaling factors to try for a given kappa map")
    parser.add_argument("--max_theta_e",    default=None,   type=float,
                        help="Maximum allowed Einstein radius, default is 35% of image fov")
    parser.add_argument("--min_theta_e",    default=None,   type=float,
                        help="Minimum allowed Einstein radius, default is 1 arcsec")

    # Physics params
    parser.add_argument("--z_source",   default=2.379,  type=float)
    parser.add_argument("--z_lens",     default=0.4457, type=float)

    args = parser.parse_args()

    distributed_strategy(args)
