import tensorflow as tf
import os, glob
import numpy as np
from astropy.constants import M_sun
from censai.data import AugmentedTNGKappaGenerator
from censai.data.kappa_tng import encode_examples
from datetime import datetime
import json


# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    kappa_files = glob.glob(os.path.join(args.kappa_dir, "*.fits"))
    if args.test_set:
        good_kappa = np.loadtxt(os.path.join(args.kappa_dir, "test_kappa.txt"))
        kappa_ids = [int(os.path.split(kap)[-1].split("_")[1]) for kap in kappa_files]
        keep_kappa = [kap_id in good_kappa for kap_id in kappa_ids]
        kappa_files = [kap_file for i, kap_file in enumerate(kappa_files) if keep_kappa[i]]
    else:
        good_kappa = np.loadtxt(os.path.join(args.kappa_dir, "train_kappa.txt"))
        kappa_ids = [int(os.path.split(kap)[-1].split("_")[1]) for kap in kappa_files]
        keep_kappa = [kap_id in good_kappa for kap_id in kappa_ids]
        kappa_files = [kap_file for i, kap_file in enumerate(kappa_files) if keep_kappa[i]]

    kappa_gen = AugmentedTNGKappaGenerator(
        kappa_fits_files=kappa_files,
        z_lens=args.z_lens,
        z_source=args.z_source,
        crop=args.crop,
        max_shift=args.max_shift,
        rotate_by=args.rotate_by,
        min_theta_e=args.min_theta_e,
        max_theta_e=args.max_theta_e,
        rescaling_size=args.rescaling_size,
        rescaling_theta_bins=args.bins
    )

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
        for i in range((THIS_WORKER - 1) * args.batch_size, args.len_dataset, N_WORKERS * args.batch_size):

            kappa, einstein_radius, rescaling_factors, kappa_ids, einstein_radius_init = kappa_gen.draw_batch(
                args.batch_size, rescale=True, shift=True, rotate=args.rotate, random_draw=args.random_draw, return_einstein_radius_init=True)

            records = encode_examples(
                kappa=kappa,
                einstein_radius_init=einstein_radius_init,
                einstein_radius=einstein_radius,
                rescalings=rescaling_factors,
                z_source=args.z_source,
                z_lens=args.z_lens,
                kappa_fov=kappa_gen.kappa_fov,
                sigma_crit=(kappa_gen.sigma_crit / (1e10 * M_sun)).decompose().value,  # 10^{10} M_sun / Mpc^2
                kappa_ids=kappa_ids
            )
            for record in records:
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",     required=True,      type=str,   help="Path to output directory")
    parser.add_argument("--len_dataset",    required=True,      type=int,   help="Size of the dataset")
    parser.add_argument("--batch_size",     default=1,          type=int,   help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--kappa_dir",      required=True,      type=str,   help="Path to directory of kappa fits files")
    parser.add_argument("--compression_type", default=None,                 help="Default is no compression. Use 'GZIP' to compress data")
    parser.add_argument("--test_set",       action="store_true",            help="Build dataset on test set of kappa maps")

    # Data generation params
    parser.add_argument("--crop",           default=0,          type=int,   help="Crop kappa map by 2*N pixels. After crop, the size of the kappa map should correspond to pixel argument "
                                                                                 "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--max_shift",      default=1.,         type=float, help="Maximum allowed shift of kappa map center in arcseconds")
    parser.add_argument("--rotate",         action="store_true",            help="Rotate the kappa map")
    parser.add_argument("--rotate_by",      default="90",                   help="'90': will rotate by a multiple of 90 degrees. 'uniform' will rotate by any angle, "
                                                                                 "with nearest neighbor interpolation and zero padding")
    parser.add_argument("--bins",           default=10,         type=int,   help="Number of bins to estimate Einstein radius distribution of a kappa given "
                                                                                 "a set of rescaling factors.")
    parser.add_argument("--random_draw",    action="store_true")
    parser.add_argument("--rescaling_size", default=100,        type=int,   help="Number of rescaling factors to try for a given kappa map")
    parser.add_argument("--max_theta_e",    default=None,       type=float, help="Maximum allowed Einstein radius, default is 5 arcseconds")
    parser.add_argument("--min_theta_e",    default=None,       type=float, help="Minimum allowed Einstein radius, default is 1 arcseconds")

    # Physics params
    parser.add_argument("--z_source",           default=1.5, type=float)
    parser.add_argument("--z_lens",             default=0.5, type=float)

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
