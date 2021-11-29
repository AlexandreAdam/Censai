from censai import RIMSharedUnetv3, PhysicalModelv2, RIMSourceUnetv2, AnalyticalPhysicalModel
from censai.models import SharedUnetModelv4, UnetModelv2
from censai.data.lenses_tng_v3 import decode_results, decode_physical_model_info
import tensorflow as tf
import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
from datetime import datetime

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

NOW = datetime.now().strftime("%y%m%d%H%M%S")


def plot_results(N, chi_squared, source, source_pred, kappa, kappa_pred, lens, lens_pred, title):
    fig, axs = plt.subplots(N, 10, figsize=(36, 4 * N))

    for i in range(N):
        axs[i, 1].imshow(source[i, ..., 0],                         cmap="bone", vmin=0, vmax=1, origin="lower")
        axs[i, 2].imshow(source_pred[i, ..., 0],                    cmap="bone", vmin=0, vmax=1, origin="lower")
        axs[i, 3].imshow(source[i, ..., 0] - source_pred[i, ..., 0],cmap="seismic", vmin=-1, vmax=1, origin="lower")

        axs[i, 4].imshow(kappa[i, ..., 0],                          cmap="hot",     norm=LogNorm(vmin=1e-1, vmax=100), origin="lower")
        axs[i, 5].imshow(kappa_pred[i, ..., 0],                     cmap="hot",     norm=LogNorm(vmin=1e-1, vmax=100), origin="lower")
        axs[i, 6].imshow(kappa[i, ..., 0] - kappa_pred[i, ..., 0],  cmap="seismic", norm=SymLogNorm(linthresh=1e-2, base=10, vmax=10, vmin=-10), origin="lower")

        axs[i, 7].imshow(lens[i, ..., 0],                           cmap="bone", vmin=0, vmax=1, origin="lower")
        axs[i, 8].imshow(lens_pred[i, ..., 0],                      cmap="bone", vmin=0, vmax=1, origin="lower")
        axs[i, 9].imshow(lens[i, ..., 0] - lens_pred[i, ..., 0],    cmap="seismic", norm=CenteredNorm(), origin="lower")

        for i in range(N):
            axs[i, 0].annotate(fr"$\chi^2 = ${chi_squared[i]:.2f}", (0.1, 0.4), textcoords="axes fraction", size=30)
            for j in range(10):
                axs[i, j].axis("off")

    fig.suptitle(f"{title}")
    axs[0, 1].set_title(r"$\mathbf{s}$")
    axs[0, 2].set_title(r"$\mathbf{\hat{s}}_T$")
    axs[0, 3].set_title(r"$\mathbf{s} - \mathbf{\hat{s}}_T$")
    axs[0, 4].set_title(r"$\kappa$")
    axs[0, 5].set_title(r"$\hat{\kappa}_T$")
    axs[0, 6].set_title(r"$\kappa - \hat{\kappa}_T$")
    axs[0, 7].set_title(r"$\mathbf{y}$")
    axs[0, 8].set_title(r"$\mathbf{\hat{y}}_T$")
    axs[0, 9].set_title(r"$\mathbf{y} - \mathbf{\hat{y}}_T$")
    return fig


def distributed_strategy(args):
    if args.minimum_date is None:
        min_date = datetime.strptime("00010100000", "%y%m%d%H%M%S")
    else:
        min_date = datetime.strptime(args.minimum_date, '%y%m%d%H%M%S')
    models = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "models", args.model_prefix + "*"))

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.train_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    train_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    # Read off global parameters from first example in dataset
    for physical_params in train_dataset.map(decode_physical_model_info):
        break
    train_dataset = train_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size)

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.val_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    val_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size)

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.test_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    test_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size)

    phys = PhysicalModelv2(
        pixels=physical_params["pixels"].numpy(),
        kappa_pixels=physical_params["kappa pixels"].numpy(),
        src_pixels=physical_params["src pixels"].numpy(),
        image_fov=physical_params["image fov"].numpy(),
        kappa_fov=physical_params["kappa fov"].numpy(),
        src_fov=physical_params["source fov"].numpy(),
        method="fft",
    )

    phys_sie = AnalyticalPhysicalModel(
        pixels=physical_params["pixels"].numpy(),
        image_fov=physical_params["image fov"].numpy(),
        src_fov=physical_params["source fov"].numpy()
    )

    # Load RIM for source only
    rim_source_dir = os.path.join(os.getenv('CENSAI_PATH'), "models", args.source_model)
    with open(os.path.join(rim_source_dir, "unet_hparams.json")) as f:
        unet_source_params = json.load(f)
    unet_source = UnetModelv2(**unet_source_params)
    with open(os.path.join(rim_source_dir, "rim_hparams.json")) as f:
        rim_source_params = json.load(f)
    rim_source = RIMSourceUnetv2(phys, unet_source, **rim_source_params)
    ckpt_s = tf.train.Checkpoint(net=unet_source)
    checkpoint_manager_s = tf.train.CheckpointManager(ckpt_s, rim_source_dir, 1)
    checkpoint_manager_s.checkpoint.restore(checkpoint_manager_s.latest_checkpoint).expect_partial()

    for model_i in range(THIS_WORKER - 1, len(models), N_WORKERS):
        model = models[model_i]
        date = datetime.strptime(model.split("_")[-1], '%y%m%d%H%M%S')
        if date < min_date:
            continue
        with open(os.path.join(model, "unet_hparams.json")) as f:
            unet_params = json.load(f)
        unet = SharedUnetModelv4(**unet_params)
        ckpt = tf.train.Checkpoint(net=unet)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, model, 1)
        checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        with open(os.path.join(model, "rim_hparams.json")) as f:
            rim_params = json.load(f)

        rim = RIMSharedUnetv3(phys, unet, **rim_params)

        dataset_names = [args.train_dataset, args.val_dataset, args.test_dataset]
        dataset_shapes = [args.train_size, args.val_size, args.test_size]
        model_name = os.path.split(model)[-1]
        for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
            data_len = dataset_shapes[i]
            dataset_name = dataset_names[i]
            for batch, (lens, source, kappa, noise_rms, psf, fwhm) in enumerate(dataset.take(data_len).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)):
                batch_size = lens.shape[0]
                # Compute predictions for kappa and source
                source_pred, kappa_pred, chi_squared = rim.predict(lens, noise_rms, psf)
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)
                # Re-optimize source with a trained source model
                source_pred2, chi_squared2 = rim_source.predict(lens, kappa_pred[-1], noise_rms, psf)
                lens_pred2 = phys.forward(source_pred2[-1], kappa_pred[-1], psf)

                N = batch_size
                fig = plot_results(N, chi_squared[-1], source, source_pred[-1], kappa, kappa_pred[-1], lens, lens_pred, title=dataset_name)
                plt.subplots_adjust(wspace=0, hspace=0)
                fig.savefig(os.path.join(os.getenv('CENSAI_PATH'), "results", f"{model_name}_{dataset_name}_{args.seed}_{NOW}.png"), bbox_inches="tight")
                plt.clf()

                fig = plot_results(N, chi_squared2[-1], source, source_pred2[-1], kappa, kappa_pred[-1], lens, lens_pred2, title=dataset_name)
                plt.subplots_adjust(wspace=0, hspace=0)
                fig.savefig(os.path.join(os.getenv('CENSAI_PATH'), "results", f"{model_name}_{args.source_model}_{dataset_name}_{args.seed}_{NOW}.png"), bbox_inches="tight")
                plt.clf()

            data_len = args.sie_size
            sie_dataset = test_dataset.take(data_len)
            for batch, (_, source, _, noise_rms, psf, fwhm) in enumerate(sie_dataset.take(data_len).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)):
                batch_size = source.shape[0]
                # Create some SIE kappa maps
                _r = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=args.max_shift)
                _theta = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=-np.pi, maxval=np.pi)
                x0 = _r * tf.math.cos(_theta)
                y0 = _r * tf.math.sin(_theta)
                ellipticity = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=args.max_ellipticity)
                phi = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=-np.pi, maxval=np.pi)
                einstein_radius = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=args.min_theta_e, maxval=args.max_theta_e)
                kappa = phys_sie.kappa_field(x0=x0, y0=y0, e=ellipticity, phi=phi, r_ein=einstein_radius)
                lens = phys.noisy_forward(source, kappa, noise_rms=noise_rms, psf=psf)

                # Compute predictions for kappa and source
                source_pred, kappa_pred, chi_squared = rim.predict(lens, noise_rms, psf)
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)
                # Re-optimize source with a trained source model
                source_pred2, chi_squared2 = rim_source.predict(lens, kappa_pred[-1], noise_rms, psf)
                lens_pred2 = phys.forward(source_pred2[-1], kappa_pred[-1], psf)

                N = batch_size
                fig = plot_results(N, chi_squared[-1], source, source_pred[-1], kappa, kappa_pred[-1], lens, lens_pred, title="SIE Test")
                plt.subplots_adjust(wspace=0, hspace=0)
                fig.savefig(os.path.join(os.getenv('CENSAI_PATH'), "results", f"{model_name}_sie_test_{args.seed}_{NOW}.png"), bbox_inches="tight")
                plt.clf()

                fig = plot_results(N, chi_squared2[-1], source, source_pred2[-1], kappa, kappa_pred[-1], lens, lens_pred2, title=dataset_name)
                plt.subplots_adjust(wspace=0, hspace=0)
                fig.savefig(os.path.join(os.getenv('CENSAI_PATH'), "results", f"{model_name}_{args.source_model}_sie_test_{args.seed}_{NOW}.png"), bbox_inches="tight")
                plt.clf()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_prefix",       required=True)
    parser.add_argument("--minimum_date",       default=None,       help="Make sure models are as recent as this date")
    parser.add_argument("--source_model",       required=True,      help="Give the path to a source RIM to converge even further the source")
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--val_dataset",        required=True,      help="Name of the dataset, not full path")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--train_dataset",      required=True)
    parser.add_argument("--train_size",         default=100,       type=int)
    parser.add_argument("--sie_size",           default=100,       type=int)
    parser.add_argument("--val_size",           default=100,        type=int)
    parser.add_argument("--test_size",          default=100,        type=int)
    parser.add_argument("--buffer_size",        default=100,        type=int)
    parser.add_argument("--batch_size",         default=10,         type=int)
    parser.add_argument("--seed",               default=None,       type=int)
    parser.add_argument("--max_shift",          default=0.1,        type=float, help="Maximum allowed shift of kappa map center in arcseconds")
    parser.add_argument("--max_ellipticity",    default=0.6,        type=float, help="Maximum ellipticty of density profile.")
    parser.add_argument("--max_theta_e",        default=0.5,        type=float, help="Maximum allowed Einstein radius")
    parser.add_argument("--min_theta_e",        default=2.5,        type=float, help="Minimum allowed Einstein radius")

    args = parser.parse_args()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    distributed_strategy(args)