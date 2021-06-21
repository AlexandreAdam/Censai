import tensorflow as tf
import os, glob
import numpy as np
from censai.data import AugmentedTNGKappaGenerator
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
    kappa_files = glob.glob(os.path.join(args.kappa_dir, "*.fits"))
    if os.path.exists(os.path.join(args.kappa_dir, "good_kappa.txt")):  # filter out bad data (see validate_kappa_maps script)
        good_kappa = np.loadtxt(os.path.join(args.kappa_dir, "good_kappa.txt"))
        kappa_ids = [int(os.path.split(kap)[-1].split("_")[1]) for kap in kappa_files]
        keep_kappa = [kap_id in good_kappa for kap_id in kappa_ids]
        kappa_files = [kap_file for i, kap_file in enumerate(kappa_files) if keep_kappa[i]]

    min_theta_e = 1 if args.min_theta_e is None else args.min_theta_e
    max_theta_e = 0.35 * args.image_fov if args.max_theta_e is None else args.max_theta_e
    kappa_gen = AugmentedTNGKappaGenerator(
        kappa_fits_files=kappa_files,
        z_lens=args.z_lens,
        z_source=args.z_source,
        crop=args.crop,
        min_theta_e=min_theta_e,
        max_theta_e=max_theta_e,
        rescaling_size=args.rescaling_size,
        rescaling_theta_bins=args.bins
    )

    phys = PhysicalModel(image_fov=args.image_fov, pixels=kappa_gen.crop_pixels,
                         kappa_fov=kappa_gen.kappa_fov, method="conv2d")

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

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"kappa_alpha_{THIS_WORKER}.tfrecords")) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch, dataset_size, N_WORKERS * args.batch):
            if args.augment:
                kappa, einstein_radius, rescaling_factors, kappa_ids = kappa_gen.draw_batch(
                    batch_size=args.batch, rescale=True, shift=args.shift, rotate=args.rotate, random_draw=False)
            else:
                kappa, einstein_radius, rescaling_factors, kappa_ids = kappa_gen.draw_batch(
                    batch_size=args.batch, rescale=False, shift=False, rotate=False, random_draw=False)

            alpha = tf.concat(phys.deflection_angle(kappa), axis=-1)

            records = encode_examples(
                kappa=kappa,
                alpha=alpha,
                rescalings=rescaling_factors,
                kappa_ids=kappa_ids,
                einstein_radius=einstein_radius
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--kappa_dir", required=True, help="Path to kappa fits files directory")
    parser.add_argument("--output_dir", required=True, help="Path where tfrecords are stored")
    parser.add_argument("--smoke_test", action="store_true")

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
    parser.add_argument("--shift", action="store_true", help="Shift center of kappa map with a budget defined by "
                                                             "the crop argument.")
    parser.add_argument("--rotate", action="store_true", help="If augment, we rotate the kappa map")
    parser.add_argument("--rotate_by", default="90",
                        help="'90': will rotate by a multiple of 90 degrees. 'uniform' will rotate by any angle, "
                             "with nearest neighbor interpolation and zero padding")
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

    # Reproducibility params
    parser.add_argument("--seed", default=None, type=int, help="Random seed for numpy and tensorflow")
    parser.add_argument("--json_override", default=None,
                        help="A json filepath that will override every command line parameters. "
                             "Useful for reproducibility")

    args = parser.parse_args()
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
