import tensorflow as tf
import os, glob
import numpy as np
from astropy.io import fits
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from censai import PhysicalModel
from censai.definitions import DTYPE, compute_rescaling_probabilities, theta_einstein, cosmo
from censai.cosmos_utils import preprocess, decode
from astropy import units as u
from astropy.constants import M_sun, c, G
from scipy.signal.windows import tukey



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
    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    n_galaxies = len(cosmos_files) * args.buffer_size
    cosmos = tf.data.TFRecordDataset(cosmos_files).map(decode).map(preprocess)
    if args.shuffle_cosmos:
        cosmos = cosmos.shuffle(buffer_size=args.buffer_size)
    cosmos = cosmos.batch(args.batch)

    # extract physical informations from fits files (common to all)
    header = fits.open(kappa_files[0])["PRIMARY"].header
    Dd = cosmo.angular_diameter_distance(args.z_lens)
    Ds = cosmo.angular_diameter_distance(args.z_source)
    Dds = cosmo.angular_diameter_distance_z1z2(args.z_lens, args.z_source)
    sigma_crit = (c**2 * Ds / (4 * np.pi * G * Dd * Dds)).to(u.kg * u.Mpc**(-2))
    # Compute a rescaling factor given possibly new redshift pair
    sigma_crit_factor = (header["SIGCRIT"] * (1e10 * M_sun * u.Mpc**(-2)) / sigma_crit).decompose().value

    pixel_scale = header["CD1_1"]  # pixel scale in arc seconds

    pixels = fits.open(kappa_files[0])["PRIMARY"].data.shape[0]  # pixels of the full cutout
    crop_pixels = pixels - 2 * args.crop                         # pixels after crop
    physical_pixel_scale = header["FOV"] / pixels * u.Mpc

    min_theta_e = 1 if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.35 * args.image_fov if args.max_theta_e is None else args.max_theta_e

    window = tukey(args.src_pixels, alpha=args.tukey_alpha)
    window = np.outer(window, window)
    phys = PhysicalModel(image_side=args.image_fov, src_side=args.source_fov, pixels=crop_pixels,
                         src_pixels=args.src_pixels, kappa_side=pixel_scale * crop_pixels, method="conv2d")

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{this_worker}.tfrecords")) as writer:
        for i in range((this_worker-1) * args.batch, args.len_dataset, N_WORKERS * args.batch):
            # for a given batch, we select unique kappa maps
            batch_indices = np.random.choice(list(range(len(kappa_files))), replace=False, size=args.batch)
            kappa = []
            for kap_index in batch_indices:
                kappa.append(fits.open(kappa_files[kap_index]))

            # select a batch of galaxies
            batch_index = np.random.randint(0, n_galaxies//args.batch)
            for galaxies, psf, ps in cosmos.skip(batch_index):  # only way to take the first batch is to fake a for loop
                break
            galaxies = window[np.newaxis, ..., np.newaxis] * galaxies

            # choose a random center shift for kappa maps, based on pixels cropped (shift by integer pixel)
            shift = np.random.randint(low=-args.crop+1, high=args.crop-1, size=(args.batch, 2))
            theta_e_init = []
            theta_e_rescaled = []
            rescalings = []
            for j in range(args.batch):
                kappa[j] = kappa[j]["PRIMARY"].data[  # crop and shift center of kappa maps
                           args.crop + shift[j, 0]: -(args.crop - shift[j, 0]),
                           args.crop + shift[j, 1]: -(args.crop - shift[j, 1])][..., np.newaxis]  # add channel dimension

                # Make sure at least a few pixels have kappa > 1 to compute Einstein radius
                if kappa[j].max() <= 1:
                    kappa[j] /= 0.95 * kappa[j].max()
                theta_e = theta_einstein(kappa[j], 1., physical_pixel_scale, sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd)[0]
                theta_e_init.append(theta_e)
                # Rough estimate of allowed rescaling factors
                rescaling_array = np.linspace(min_theta_e/theta_e, max_theta_e/theta_e, args.rescaling_size) * sigma_crit_factor
                # compute probability distribution of rescaling so that theta_e ~ Uniform(min_theta_e, max_theta_e)
                rescaling_p = compute_rescaling_probabilities(kappa[j], rescaling_array, physical_pixel_scale,
                                                              sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd,
                                                              bins=args.bins, min_theta_e=min_theta_e, max_theta_e=max_theta_e)
                # make an informed random choice
                rescaling = np.random.choice(rescaling_array, size=1, p=rescaling_p)[0]
                # rescale
                kappa[j] = rescaling * kappa[j]
                theta_e_rescaled.append(theta_einstein(kappa[j], 1., physical_pixel_scale, sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd)[0])
                rescalings.append(rescaling)
            kappa = tf.stack(kappa, axis=0)
            kappa = tf.cast(kappa, dtype=DTYPE)

            lensed_images = phys.noisy_forward(galaxies, kappa, noise_rms=args.noise_rms)

            for j in range(args.batch):
                features = {
                    "kappa": _bytes_feature(kappa[j].numpy().tobytes()),
                    "source": _bytes_feature(galaxies[j].numpy().tobytes()),
                    "lens": _bytes_feature(lensed_images[j].numpy().tobytes()),
                    "Einstein radius before rescaling": _float_feature(theta_e_init[j]),
                    "Einstein radius": _float_feature(theta_e_rescaled[j]),
                    "rescaling factor": _float_feature(rescalings[j]),
                    "power spectrum": _bytes_feature(ps[j].numpy().tobytes()),
                    "z source": _float_feature(args.z_source),
                    "z lens": _float_feature(args.z_lens),
                    "image fov": _float_feature(args.image_fov),             # arcsec
                    "kappa fov": _float_feature(pixel_scale * crop_pixels),  # arcsec
                    "sigma crit": _float_feature(                            # 10^{10} M_sun / Mpc^2
                        (sigma_crit / (1e10*M_sun)).decompose().to(u.Mpc**(-2)).value
                    ),
                    "src pixels": _int64_feature(args.src_pixels),
                    "kappa pixels": _int64_feature(crop_pixels),
                    "noise rms": _float_feature(args.noise_rms)
                }

                serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
                record = serialized_output.SerializeToString()
                writer.write(record)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help="Path to output directory")
    parser.add_argument("--len_dataset", required=True, type=int, help="Size of the dataset")
    parser.add_argument("--kappa_dir", required=True, type=str, help="Path to directory of kappa fits files")
    parser.add_argument("--cosmos_dir", required=True, type=str,
                        help="Path to directory of galaxy brightness distribution tfrecords "
                             "(output of cosmos_to_tfrecors.py)")
    # Physical model params
    parser.add_argument("--src_pixels", default=128, type=int, help="Size of Cosmos postage stamps")
    parser.add_argument("--image_fov", default=20, type=float,
                        help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--source_fov", default=3, type=float,
                        help="Field of view of the source plane in arc seconds")
    parser.add_argument("--noise_rms", default=0.3e-3, type=float,
                        help="White noise RMS added to lensed image")
    #TODO add an option to change color of the noise to match COSMOS color

    # Data generation params
    parser.add_argument("--crop", default=0, type=int,
                        help="Crop kappa map by 2*N pixels. After crop, the size of the kappa map "
                             "should correspond to pixel argument "
                             "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--shuffle_cosmos", action="store_true", help="Shuffle indices of cosmos dataset")
    parser.add_argument("--buffer_size", default=1000, type=int,
                        help="Should match example_per_shard when tfrecords were produced "
                             "(only used if shuffle_cosmos is called)")
    parser.add_argument("--batch", default=1, type=int,
                        help="Number of examples worked out in a s ingle pass by a worker")
    parser.add_argument("--tukey_alpha", default=0.6, type=float,  # help from scipy own documentation
                        help="Shape parameter of the Tukey window, representing the fraction of the "
                             "window inside the cosine tapered region. "
                             "If 0, the Tukey window is equivalent to a rectangular window. "
                             "If 1, the Tukey window is equivalent to a Hann window. "
                             "This window is used on cosmos postage stamps.")
    parser.add_argument("--bins", default=10, type=int,
                        help="Number of bins to estimate Einstein radius distribution of a kappa given "
                             "a set of rescaling factors.")
    parser.add_argument("--rescaling_size", default=100, type=int,
                        help="Number of rescaling factors to try for a given kappa map")
    parser.add_argument("--max_theta_e", default=None, type=float,
                        help="Maximum allowed Einstein radius, default is 35% of image fov")
    parser.add_argument("--min_theta_e", default=None, type=float,
                        help="Minimum allowed Einstein radius, default is 1 arcsec")

    # Physics params
    parser.add_argument("--z_source", default=2.379, type=float)
    parser.add_argument("--z_lens", default=0.4457, type=float)

    args = parser.parse_args()

    distributed_strategy(args)
