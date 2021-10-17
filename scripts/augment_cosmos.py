import tensorflow as tf
import os, glob
import numpy as np
from censai.utils import _bytes_feature, _int64_feature
from censai.data.cosmos import preprocess_image as preprocess, decode_image as decode
from datetime import datetime
import json
import time

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    cosmos_files = glob.glob(os.path.join(args.cosmos_dir, "*.tfrecords"))
    cosmos = tf.data.TFRecordDataset(cosmos_files)
    cosmos = cosmos.map(decode).map(preprocess).batch(args.batch_size)

    max_shift = min(args.crop, args.max_shift)

    options = tf.io.TFRecordOptions(compression_type=args.compression_type)
    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data.tfrecords"), options) as writer:
        for galaxies in cosmos:
            for j in range(galaxies.shape[0]):
                angle = np.random.randint(low=0, high=3, size=1)[0]
                galaxy = tf.image.rot90(galaxies[j], k=angle).numpy()
                if args.crop > 0:
                    shift = np.random.randint(low=-max_shift, high=max_shift, size=2)
                    galaxy = galaxy[args.crop + shift[0]: -(args.crop - shift[0]), args.crop + shift[1]: -(args.crop - shift[1]), ...]

                features = {
                    "image": _bytes_feature(galaxy.tobytes()),
                    "height": _int64_feature(galaxy.shape[0]),
                }
                record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                writer.write(record)
    print(f"Finished work at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir",       required=True,      type=str,   help="Path to output directory")
    parser.add_argument("--batch_size",       default=1,          type=int,   help="Number of examples worked out in a single pass by a worker")
    parser.add_argument("--cosmos_dir",       required=True,      type=str,   help="Path to directory of galaxy brightness distribution tfrecords (output of cosmos_to_tfrecors.py)")
    parser.add_argument("--compression_type", default=None,                   help="Default is no compression. Use 'GZIP' to compress data")

    # Data generation params
    parser.add_argument("--crop",           default=0,          type=int,   help="Crop kappa map by 2*N pixels. After crop, the size of the kappa map should correspond to pixel argument "
                                                                                 "(e.g. kappa of 612 pixels cropped by N=50 on each side -> 512 pixels)")
    parser.add_argument("--max_shift",      default=0,          type=int,   help="Maximum number of pixels by which to shift the image. In any case, shift will be < crop.")

    args = parser.parse_args()
    if THIS_WORKER > 1:
        time.sleep(5)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with open(os.path.join(args.output_dir, "script_params.json"), "w") as f:
        args_dict = vars(args)
        json.dump(args_dict, f)

    distributed_strategy(args)
