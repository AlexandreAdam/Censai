import tensorflow as tf
from censai import PhysicalModel, RIMSharedUnet, PowerSpectrum
from censai.models import SharedUnetModel
from argparse import Namespace
from censai.definitions import log_10
import os, glob, json
import h5py
import numpy as np


def main(args):
    if args.v1:
        from censai.data.lenses_tng import decode_train, decode_physical_model_info, preprocess
    else:
        from censai.data.lenses_tng_v2 import decode_train, decode_physical_model_info, preprocess

    files = glob.glob(os.path.join(os.getenv("CENSAI_PATH"), "data", args.dataset, "*.tfrecords"))
    # Read concurrently from multiple records
    files = tf.data.Dataset.from_tensor_slices(files)
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type),
                               block_length=args.block_length, num_parallel_calls=tf.data.AUTOTUNE)
    # Read off global parameters from first example in dataset
    for physical_params in dataset.map(decode_physical_model_info):
        break
    # preprocessing
    dataset = dataset.map(decode_train).map(preprocess)

    checkpoints_dir = os.path.join(os.getenv("CENSAI_PATH"), "models", args.model)

    with open(os.path.join(checkpoints_dir, "script_params.json"), "r") as f:
        run_args = json.load(f)
    run_args = Namespace(**run_args)
    
    phys = PhysicalModel(
        pixels=physical_params["pixels"].numpy(),
        kappa_pixels=physical_params["kappa pixels"].numpy(),
        src_pixels=physical_params["src pixels"].numpy(),
        image_fov=physical_params["image fov"].numpy(),
        kappa_fov=physical_params["kappa fov"].numpy(),
        src_fov=physical_params["source fov"].numpy(),
        method=args.forward_method,
        noise_rms=physical_params["noise rms"].numpy(),
        psf_sigma=physical_params["psf sigma"].numpy()
    )

    unet = SharedUnetModel(
        filters=run_args.filters,
        filter_scaling=run_args.filter_scaling,
        kernel_size=run_args.kernel_size,
        layers=run_args.layers,
        block_conv_layers=run_args.block_conv_layers,
        strides=run_args.strides,
        bottleneck_kernel_size=run_args.bottleneck_kernel_size,
        bottleneck_filters=run_args.bottleneck_filters,
        resampling_kernel_size=run_args.resampling_kernel_size,
        input_kernel_size=run_args.input_kernel_size,
        gru_kernel_size=run_args.gru_kernel_size,
        upsampling_interpolation=run_args.upsampling_interpolation,
        activation=run_args.activation,
        alpha=run_args.alpha,
        batch_norm=run_args.batch_norm,
        dropout_rate=run_args.dropout_rate,
        gru_architecture=run_args.gru_architecture
    )
    unet.trainable = False
    rim = RIMSharedUnet(
        physical_model=phys,
        unet=unet,
        steps=run_args.steps,
        adam=run_args.adam,
        kappalog=run_args.kappalog,
        source_link=run_args.source_link,
        kappa_normalize=run_args.kappa_normalize,
        kappa_init=run_args.kappa_init,
        source_init=run_args.source_init
    )

    ckpt = tf.train.Checkpoint(net=rim.unet)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, args.rim, max_to_keep=1)
    checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()

    output_dir = os.path.join(os.getenv("CENSAI_PATH"), "results", args.model + "_" + args.dataset)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    ps_lens = PowerSpectrum(bins=args.lens_k_bins, pixels=physical_params["pixels"].numpy())
    ps_x = PowerSpectrum(bins=args.x_k_bins,  pixels=physical_params["kappa pixels"].numpy())

    shards = args.total_items // args.examples_per_shard + 1 * (args.total_items % args.examples_per_shard > 0)
    k = 0
    for shard in range(shards):
        hf = h5py.File(os.path.join(output_dir, f"predictions_{shard:02d}.h5"), 'w')
        data = dataset.skip(shard * args.examples_per_shard).take(args.examples_per_shard).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for batch, (lens, source, kappa) in enumerate(data):
            source_pred, kappa_pred, chi_squared = rim.predict(lens)
            lens_pred = phys.forward(source_pred[-1], kappa_pred[-1])
            lam = phys.lagrange_multiplier(y_true=lens, y_pred=lens_pred)

            # remove channel dimension because power spectrum expect only [batch, pixels, pixels] shaped tensor.
            _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
            _ps_kappa = ps_x.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
            _ps_source = ps_x.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])

            batch_size = lens.shape[0]
            for b in range(batch_size):
                g = hf.create_group(f'data_{k:d}')
                g.create_dataset("lens",        data=lens[b])
                g.create_dataset("source",      data=source[b])
                g.create_dataset("kappa",       data=kappa[b])
                g.create_dataset("lens_pred",   data=lens_pred[b])
                g.create_dataset("source_pred", data=source_pred[:, b])
                g.create_dataset("kappa_pred",  data=kappa_pred[:, b])
                g.create_dataset("chi_squared", data=chi_squared[:, b])
                g.create_dataset("lambda",      data=lam[b])
                g.create_dataset("ps_lens",     data=_ps_lens[b])
                g.create_dataset("ps_kappa",    data=_ps_kappa[b])
                g.create_dataset("ps_source",   data=_ps_source[b])
                k += 1
        hf.close()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset",           required=True,                       help="Name of dataset")
    parser.add_argument("--total_items",        required=True,      type=int)
    parser.add_argument("--forward_method",     default="conv2d",                   help="Physical method to compute deflection angles. conv2d or fft")
    parser.add_argument("--compression_type",   default="GZIP",                     help="tfrecords compression type of the datasets. Default is GZIP.")
    parser.add_argument("--block_length",       default=1,          type=int)
    parser.add_argument("--batch_size",         default=1,          type=int)
    parser.add_argument("--examples_per_shard", default=1000,       type=int,       help="Results are saved in hdf5 format, in shards of n examples.")
    parser.add_argument("--model",              required=True,                      help="Name of the model only, not a path since we get it from $CENSAI_PATH/models/")
    parser.add_argument("--v1",                 action="store_true",                help="By default, use version 2 to decode data")
    parser.add_argument("--x_k_bins",           default=40,         type=int,       help="Number of bins to use for the power spectrum of kappa and source.")
    parser.add_argument("--lens_k_bins",        default=40,         type=int,       help="Number of bins to use for the power spectrum of the lens.")
