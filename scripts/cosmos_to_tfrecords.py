import tensorflow as tf
import os
from censai.data.cosmos import encode_examples
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--pixel_scale",        default=0.03,       type=float, help="Native pixel resolution of the image")
parser.add_argument("--img_len",            default=128,        type=int,   help="Number of pixels on a side of the drawn postage stamp")
parser.add_argument("--example_per_shard",  default=1000,       type=int,   help="Number of example on a given COSMO shard")
parser.add_argument("--task_id",            default=-1,         type=int,   help="Id of the task (1-50 for default 25.2 dataset)")
parser.add_argument("--do_all",             action="store_true",            help="Override task id and do and tasks")
parser.add_argument("--sample",             default="25.2",                 help="Either 25.2 or 23.5")
parser.add_argument("--exclusion_level",    default="marginal",             help="Galsim exclusion level of bad postage stamps")
parser.add_argument("--cosmos_dir",         default=None,                   help="Directory to cosmos data")
parser.add_argument("--store_attributes",   action="store_true",            help="Wether to store ['mag_auto', 'flux_radius', 'sersic_n', 'sersic_q', 'z_phot] or not")
parser.add_argument("--rotation",           action="store_true",            help="Rotate randomly the postage stamp (and psf)")
parser.add_argument("--output_dir",         required=True,                  help="Path to the directory where to store tf records")
parser.add_argument("--compression_type",   default=None,                   help="Default is no compression. Use 'GZIP' to compress")

args = parser.parse_args()
if args.store_attributes:
    vars(args)["attributes"] = ['mag_auto', 'flux_radius', 'sersic_n', 'sersic_q', 'zphot']
options = tf.io.TFRecordOptions(compression_type=args.compression_type)

if args.do_all:
    if args.sample == "23.5":
        n_tasks = 58
    elif args.sample == "25.2":
        n_tasks = 88
    else:
        raise NotImplementedError
    for task_id in range(0, n_tasks):
        gen = encode_examples(args, task_id=task_id, sample=args.sample, cosmos_dir=args.cosmos_dir,
                              exclusion_level=args.exclusion_level)
        filename = os.path.join(args.output_dir, f"cosmos_record_{task_id}.tfrecords")
        with tf.io.TFRecordWriter(filename, options=options) as writer:
            for record in gen:
                writer.write(record)
else:
    gen = encode_examples(args, task_id=args.task_id, sample=args.sample, cosmos_dir=args.cosmos_dir,
                          exclusion_level=args.exclusion_level)
    filename = os.path.join(args.output_dir, f"cosmos_record_{args.task_id}.tfrecords")
    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for record in gen:
            writer.write(record)
