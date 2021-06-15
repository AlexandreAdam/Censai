import tensorflow as tf
import os, glob
import numpy as np
from astropy.io import fits
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from censai import PhysicalModel
from censai.cosmos_utils import preprocess, decode


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
    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    cosmos = tf.data.TFRecordDataset(cosmos_files).map(decode).map(preprocess)
    if args.shuffle_cosmos:
        cosmos = cosmos.shuffle(buffer_size=args.shuffle_buffer_size)
    cosmos = cosmos.batch(args.batch)

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{this_worker}.tfrecords")) as writer:
        for i in range((this_worker-1) * args.batch, args.len_dataset, N_WORKERS * args.batch):
            # for a given batch, we select unique kappa maps
            batch_indices = np.random.choice(list(range(len(kappa_files))), replace=False, size=args.batch)
            kappa = []
            for file in kappa_files[batch_indices]:
                kappa.append(fits.open(file))

            for j in range(args.batch):
                features = {
                }

                serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
                record = serialized_output.SerializeToString()
                writer.write(record)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=str, help="Path to output directory")
    parser.add_argument("--len_dataset", required=True, type=str, help="Size of the dataset")
    parser.add_argument("--kappa_dir", required=True, type=str, help="Path to directory of kappa fits files")
    parser.add_argument("--cosmos_dir", required=True, type=str,
                        help="Path to directory of galaxy brightness distribution tfrecords "
                             "(output of cosmos_to_tfrecors.py)")
    parser.add_argument("--pixels", default=512, type=int, help="Number of pixels on the side of a square array")
    parser.add_argument("--crop", default=0, type=int,
                        help="Crop kappa map by 2*N pixels. After crop, the size of the kappa map "
                             "should correspond to pixel argument "
                             "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--shuffle_cosmos", action="store_true", help="Shuffle indices of cosmos dataset")
    parser.add_argument("--shuffle_buffer_size", default=1000, type=int,
                        help="Should match example_per_shard when tfrecords were produced "
                             "(only used if shuffle_cosmos is called)")
    parser.add_argument("--batch", default=1, type=int,
                        help="Number of examples worked out in a single pass by a worker")

    args = parser.parse_args()

    distributed_strategy(args)
