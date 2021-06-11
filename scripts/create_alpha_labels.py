import tensorflow as tf
import os, glob
import numpy as np
from astropy.io import fits
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from censai import PhysicalModel
from argparse import ArgumentParser

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    phys = PhysicalModel(image_side=args.image_fov, src_side=args.source_fov, pixels=args.pixels,
                         kappa_side=args.kappa_fov, method="conv2d")
    kappa_files = glob.glob(os.path.join(args.kappa_dir, "*.fits"))
    if args.smoke_test:
        kappa_files = kappa_files[:N_WORKERS*args.batch]
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"kappa_alpha_{this_worker}.tfrecords")) as writer:
        for i in range(this_worker-1 * args.batch, len(kappa_files), N_WORKERS * args.batch):
            files = kappa_files[i: i + args.batch]
            filenames = [os.path.split(file)[-1] for file in files]
            kappa_ids = [int(filename.split("_")[1]) for filename in filenames]  # format is kappa_{id}_xy.fits
            kappa_fits = [fits.open(file) for file in files]
            # add missing batch and channel dimension to kappa map, then stack them along batch dim
            kappa = np.concatenate([kap["PRIMARY"].data[np.newaxis, ..., np.newaxis] for kap in kappa_fits], axis=0)
            kappa = kappa[..., args.crop:args.pixels-args.crop, args.crop:args.pixels-args.crop, :]
            if args.augment:
                factors = 1 + np.random.exponential(args.exponential_rate/kappa.max(axis=(1, 2, 3)))
                kappa *= factors[..., np.newaxis, np.newaxis, np.newaxis]
            alpha = tf.concat(phys.deflection_angle(kappa), axis=-1).numpy()  # compute labels here, bring back to numpy

            for j in range(args.batch):
                features = {
                        "kappa": _bytes_feature(kappa[j].tobytes()),
                        "pixels": _int64_feature(args.pixels),
                        "alpha": _bytes_feature(alpha[j].tobytes()),
                        "rescale": _float_feature(factors[j]),
                        "sigma_crit": _float_feature(kappa_fits[j][0].header["SIGCRIT"]),
                        "kappa_id": _int64_feature(kappa_ids[j])
                    }

                serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
                record = serialized_output.SerializeToString()
                writer.write(record)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kappa_dir", required=True, help="Path to kappa fits files directory")
    parser.add_argument("--image_fov", default=16, type=float,
                        help="Field of view of the image (lens plane) in arc seconds")
    parser.add_argument("--source_fov", default=3, type=float,
                        help="Field of view of the source plane in arc seconds")
    parser.add_argument("--pixels", default=512, type=int,
                        help="Number of pixels on a side of the image in the lens plane")
    parser.add_argument("--kappa_fov", default=16, type=int,
                        help="Field of view of the kappa map")
    parser.add_argument("--batch", default=1, type=int,
                        help="Number of label maps to be computed at the same time")
    parser.add_argument("--crop", default=0, type=int,
                        help="Crop kappa map by N pixels. After crop, the size of the kappa map "
                             "should correspond to pixel argument "
                             "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--augment", action="store_true",
                        help="Rescale kappa map by a random number, which is stored in the record for future reference.")
    parser.add_argument("--exponential_rate", default=1, type=float,
                        help="If we augment data, what is the base rate (ie when max(kappa)=1) "
                             "of the exponential distribution (factor = 1 + Exponential(rate/max(kappa))")
    parser.add_argument("--output_dir", required=True, help="Path where tfrecords are stored")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    distributed_strategy(args)
