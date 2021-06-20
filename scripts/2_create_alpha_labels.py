import tensorflow as tf
import os, glob
import numpy as np
from astropy.io import fits
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from censai import PhysicalModel
from censai.definitions import cosmo, theta_einstein, compute_rescaling_probabilities, DTYPE
from astropy.constants import M_sun, c, G
from astropy import units as u
from argparse import ArgumentParser
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

    # extract physical informations from fits files (common to all)
    header = fits.open(kappa_files[0])["PRIMARY"].header
    Dd = cosmo.angular_diameter_distance(args.z_lens)
    Ds = cosmo.angular_diameter_distance(args.z_source)
    Dds = cosmo.angular_diameter_distance_z1z2(args.z_lens, args.z_source)
    sigma_crit = (c ** 2 * Ds / (4 * np.pi * G * Dd * Dds)).to(u.kg * u.Mpc ** (-2))
    # Compute a rescaling factor given possibly new redshift pair
    sigma_crit_factor = (header["SIGCRIT"] * (1e10 * M_sun * u.Mpc ** (-2)) / sigma_crit).decompose().value

    pixel_scale = header["CD1_1"]  # pixel scale in arc seconds

    pixels = fits.open(kappa_files[0])["PRIMARY"].data.shape[0]  # pixels of the full cutout
    crop_pixels = pixels - 2 * args.crop  # pixels after crop
    physical_pixel_scale = header["FOV"] / pixels * u.Mpc

    min_theta_e = 1 if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.35 * args.image_fov if args.max_theta_e is None else args.max_theta_e

    phys = PhysicalModel(image_fov=args.image_fov, pixels=crop_pixels,
                         kappa_fov=pixel_scale * crop_pixels, method="conv2d")

    if args.smoke_test:
        kappa_files = kappa_files[:N_WORKERS*args.batch]

    if args.augment:
        dataset_size = int((1 + args.augment) * len(kappa_files))
        _original_len = len(kappa_files)
        for i in range(int(args.augment + 1)):
            # make copies of the dataset so its length >= the augmented dataset size
            kappa_files = kappa_files + kappa_files[:_original_len]
    else:
        dataset_size = len(kappa_files)

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"kappa_alpha_{this_worker}.tfrecords")) as writer:
        print(f"Started worker {this_worker} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((this_worker-1) * args.batch, dataset_size, N_WORKERS * args.batch):
            files = kappa_files[i: i + args.batch]
            kappa = []
            for file in files:
                kappa.append(fits.open(file))
            kappa_ids = [kap["PRIMARY"].header["SUBID"] for kap in kappa]
            kappa = [kap["PRIMARY"].data[..., np.newaxis] for kap in kappa]  # add channel dim

            if args.augment:
                # choose a random center shift for kappa maps, based on pixels cropped (shift by integer pixel)
                if args.crop:
                    shift = np.random.randint(low=-args.crop + 1, high=args.crop - 1, size=(args.batch, 2))
                theta_e_init = []
                theta_e_rescaled = []
                rescalings = []
                for j in range(args.batch):
                    if args.crop:
                        kappa[j] = kappa[j][  # crop and shift center of kappa maps
                                   args.crop + shift[j, 0]: -(args.crop - shift[j, 0]),
                                   args.crop + shift[j, 1]: -(args.crop - shift[j, 1]), ...]

                    # Make sure at least a few pixels have kappa > 1
                    if kappa[j].max() <= 1:
                        kappa[j] /= 0.95 * kappa[j].max()
                    theta_e = theta_einstein(kappa[j], 1., physical_pixel_scale, sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd)[0]
                    theta_e_init.append(theta_e)
                    # Rough estimate of allowed rescaling factors
                    rescaling_array = np.linspace(min_theta_e / theta_e, max_theta_e / theta_e, args.rescaling_size) * sigma_crit_factor
                    # compute probability distribution of rescaling so that theta_e ~ Uniform(min_theta_e, max_theta_e)
                    rescaling_p = compute_rescaling_probabilities(kappa[j], rescaling_array, physical_pixel_scale,
                                                                  sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd,
                                                                  bins=args.bins, min_theta_e=min_theta_e,
                                                                  max_theta_e=max_theta_e)
                    if rescaling_p.sum() == 0:
                        rescaling = 1.
                    else:
                        # make an informed random choice
                        rescaling = np.random.choice(rescaling_array, size=1, p=rescaling_p)[0]
                    # rescale
                    kappa[j] = rescaling * kappa[j]
                    theta_e_rescaled.append(
                        theta_einstein(kappa[j], 1., physical_pixel_scale, sigma_crit, Dds=Dds, Ds=Ds, Dd=Dd)[0])
                    rescalings.append(rescaling)
            elif args.crop:
                kappa = [kap[args.crop: -args.crop, args.crop: -args.crop, ...] for kap in kappa]
                rescalings = [1.] * args.batch
            else:
                rescalings = [1.] * args.batch
            kappa = tf.stack(kappa, axis=0)
            kappa = tf.cast(kappa, dtype=DTYPE)
            alpha = tf.concat(phys.deflection_angle(kappa), axis=-1)  # compute labels here

            for j in range(args.batch):
                features = {
                        "kappa": _bytes_feature(kappa[j].numpy().tobytes()),
                        "pixels": _int64_feature(crop_pixels),
                        "alpha": _bytes_feature(alpha[j].numpy().tobytes()),
                        "rescale": _float_feature(rescalings[j]),
                        "kappa_id": _int64_feature(kappa_ids[j]),
                        "Einstein radius": _float_feature(theta_e_rescaled[j])
                    }

                serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
                record = serialized_output.SerializeToString()
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kappa_dir", required=True, help="Path to kappa fits files directory")

    # Physical Model params
    parser.add_argument("--image_fov", default=20, type=float,
                        help="Field of view of the image (lens plane) in arc seconds")

    # Data generation params
    parser.add_argument("--batch", default=1, type=int,
                        help="Number of label maps to be computed at the same time")
    parser.add_argument("--crop", default=0, type=int,
                        help="Crop kappa map by N pixels. After crop, the size of the kappa map "
                             "should correspond to pixel argument "
                             "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--augment", default=0., type=float,
                        help="Percentage by which to augment the data. (0=no augmentation)"
                             "Random shift of the position of the "
                             "kappa maps if crops > 0, and random rescaling.")
    parser.add_argument("--bins", default=10, type=int,
                        help="Number of bins to estimate Einstein radius distribution of a kappa given "
                             "a set of rescaling factors.")
    parser.add_argument("--rescaling_size", default=100, type=int,
                        help="Number of rescaling factors to try for a given kappa map")
    parser.add_argument("--max_theta_e", default=None, type=float,
                        help="Maximum allowed Einstein radius, default is 35% of image fov")
    parser.add_argument("--min_theta_e", default=None, type=float,
                        help="Minimum allowed Einstein radius, default is 1 arcsec")

    parser.add_argument("--output_dir", required=True, help="Path where tfrecords are stored")
    parser.add_argument("--smoke_test", action="store_true")

    # Physics params
    parser.add_argument("--z_source", default=2.379, type=float)
    parser.add_argument("--z_lens", default=0.4457, type=float)

    args = parser.parse_args()

    distributed_strategy(args)
