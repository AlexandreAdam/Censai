from censai.models import VAE
from censai.utils import reconstruction_plot, sampling_plot
import numpy as np
import tensorflow as tf
import os, glob
import json


def main(args):
    files = []
    files.extend(glob.glob(os.path.join(args.dataset, "*.tfrecords")))
    np.random.shuffle(files)
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)

    if args.type == "cosmos":
        from censai.data.cosmos import decode_shape, decode_image as decode, preprocess_image as preprocess
    elif args.type == "kappa":
        from censai.data.kappa_tng import decode_shape, decode_train as decode
        from censai.definitions import log_10 as preprocess
    # Read off global parameters from first example in dataset
    for pixels in dataset.map(decode_shape):
        break
    vars(args).update({"pixels": int(pixels)})
    dataset = dataset.map(decode).map(preprocess).batch(args.batch_size)

    model_list = glob.glob(os.path.join(os.getenv("CENSAI_PATH"), "models", args.model_prefixe, "*"))
    for model in model_list:
        if "second_stage" in model:
            continue
        with open(os.path.join(model, "model_hparams.json")) as f:
            vae_hparams = json.load(f)

        # load weights
        vae = VAE(**vae_hparams)
        ckpt1 = tf.train.Checkpoint(net=vae)
        checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, model, 1)
        checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

        for batch, images in enumerate(dataset):
            y_pred = vae(images)
            fig = reconstruction_plot(images, y_pred)
            fig.savefig(os.path.join(os.getenv("CENSAI_PATH"), "results", "vae_reconstruction" + model + args.output_postfixe + f"{batch:01d}.png"))
            y_pred = vae.sample(args.sampling_size)
            fig = sampling_plot(y_pred)
            fig.savefig(os.path.join(os.getenv("CENSAI_PATH"), "results", "vae_sampling" + model + args.output_postfixe + f"{batch:01d}.png"))

            if batch == args.n_plots-1:
                break


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_prefixe",      required=True)
    parser.add_argument("--dataset",            required=True)
    parser.add_argument("--type",               required=True,      help="One of ['kappa', 'cosmos']")
    parser.add_argument("--output_postfixe",    default="",         help="phrase to append to result filename")
    parser.add_argument("--batch_size",         default=20,         type=int,          help="Number of rows for a single reconstruction plot")
    parser.add_argument("--sampling_size",      default=81,         type=int,          help="Number of images to sample for a single figure")
    parser.add_argument("--n_plots",            default=1,          type=int,          help="Number of plot to make for a given model")
    parser.add_argument("--block_length",       default=1,          type=int,          help="Number of example to read from tfrecords concurently")
    parser.add_argument("--compression_type",   default=None,                           help="Compression type used to write data. Default assumes no compression.")
    parser.add_argument("--seed",               default=None,       type=int)

    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    main(args)
