from censai import RIMSharedUnetv3, PhysicalModelv2, PowerSpectrum, RIMSourceUnetv2
from censai.models import SharedUnetModelv4, UnetModelv2
from censai.data.lenses_tng_v3 import decode_results, decode_physical_model_info
import tensorflow as tf
import numpy as np
import os, glob, json
import h5py
from censai.definitions import log_10

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    models = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "models", args.model_prefix + "*"))

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.train_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    train_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    # Read off global parameters from first example in dataset
    for physical_params in train_dataset.map(decode_physical_model_info):
        break
    train_dataset = train_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size).take(args.total_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.val_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    val_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size).take(args.val_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.test_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    test_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(decode_results).shuffle(buffer_size=args.buffer_size).take(args.val_size).batch(args.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    ps_lens = PowerSpectrum(bins=args.lens_coherence_bins, pixels=physical_params["pixels"].numpy())
    ps_source = PowerSpectrum(bins=args.source_coherence_bins,  pixels=physical_params["source pixels"].numpy())
    ps_kappa = PowerSpectrum(bins=args.kappa_coherence_bins,  pixels=physical_params["kappa pixels"].numpy())

    phys = PhysicalModelv2(
        pixels=physical_params["pixels"].numpy(),
        kappa_pixels=physical_params["kappa pixels"].numpy(),
        src_pixels=physical_params["src pixels"].numpy(),
        image_fov=physical_params["image fov"].numpy(),
        kappa_fov=physical_params["kappa fov"].numpy(),
        src_fov=physical_params["source fov"].numpy(),
        method="fft",
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
        with open(os.path.join(model, "unet_hparams.json", "r")) as f:
            unet_params = json.load(f)
        unet = SharedUnetModelv4(**unet_params)
        ckpt = tf.train.Checkpoint(net=unet)
        checkpoint_manager = tf.train.CheckpointManager(ckpt, model, 1)
        checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        with open(os.path.join(model, "rim_hparams.json", "r")) as f:
            rim_params = json.load(f)

        rim = RIMSharedUnetv3(phys, unet, **rim_params)

        dataset_names = [args.train_dataset, args.val_dataset, args.test_dataset]
        dataset_shapes = [args.train_size, args.val_size, args.test_size]
        model_name = os.path.split(model)[-1]
        with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", model_name + "_" + args.source_model + ".h5"), 'w') as hf:
            for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
                g = hf.create_group(f'{dataset_names[i]}')
                data_len = dataset_shapes[i]
                g.create_dataset(name="lens", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
                g.create_dataset(name="psf",  shape=[data_len, physical_params['psf pixels'], physical_params['psf pixels'], 1], dtype=np.float32)
                g.create_dataset(name="psf_fwhm", shape=[data_len], dtype=np.float32)
                g.create_dataset(name="noise_rms", shape=[data_len], dtype=np.float32)
                g.create_dataset(name="source", shape=[data_len, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
                g.create_dataset(name="kappa", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
                g.create_dataset(name="lens_pred", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
                g.create_dataset(name="lens_pred2", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
                g.create_dataset(name="source_pred", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
                g.create_dataset(name="source_pred2", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
                g.create_dataset(name="kappa_pred", shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
                g.create_dataset(name="chi_squared", shape=[data_len, rim.steps], dtype=np.float32)
                g.create_dataset(name="chi_squared2", shape=[data_len, rim.steps], dtype=np.float32)
                g.create_dataset(name="lens_coherence_spectrum", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
                g.create_dataset(name="source_coherence_spectrum",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
                g.create_dataset(name="lens_coherence_spectrum2", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
                g.create_dataset(name="source_coherence_spectrum2",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
                g.create_dataset(name="kappa_coherence_spectrum", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
                g.create_dataset(name="lens_frequencies", shape=[args.lens_coherence_bins], dtype=np.float32)
                g.create_dataset(name="source_frequencies", shape=[args.source_coherence_bins], dtype=np.float32)
                g.create_dataset(name="kappa_frequencies", shape=[args.kappa_coherence_bins], dtype=np.float32)
                g.create_dataset(name="kappa_fov", shape=[1], dtype=np.float32)
                g.create_dataset(name="source_fov", shape=[1], dtype=np.float32)
                g.create_dataset(name="lens_fov", shape=[1], dtype=np.float32)

                for batch, (lens, source, kappa, noise_rms, psf, fwhm) in dataset:
                    # Compute predictions for kappa and source
                    source_pred, kappa_pred, chi_squared = rim.predict(lens, noise_rms, psf)
                    lens_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)
                    # Re-optimize source with a trained source model
                    source_pred2, chi_squared2 = rim_source.predict(lens, kappa_pred[-1], noise_rms, psf)
                    lens_pred2 = phys.forward(source_pred2[-1], kappa_pred[-1], psf)
                    # Compute Power spectrum of converged predictions
                    _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
                    _ps_lens2 = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred2[..., 0])
                    _ps_kappa = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
                    _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
                    _ps_source2 = ps_source.cross_correlation_coefficient(source[..., 0], source_pred2[-1][..., 0])

                    # save results
                    i_begin = batch * args.batch_size
                    i_end = i_begin + args.batch_size
                    g["lens"][i_begin:i_end] = lens.numpy().astype(np.float32)
                    g["psf"][i_begin:i_end] = psf.numpy().astype(np.float32)
                    g["psf_fwhm"][i_begin:i_end] = fwhm.numpy().astype(np.float32)
                    g["noise_rms"][i_begin:i_end] = noise_rms.numpy().astype(np.float32)
                    g["source"][i_begin:i_end] = source.numpy().astype(np.float32)
                    g["kappa"][i_begin:i_end] = kappa.numpy().astype(np.float32)
                    g["lens_pred"][i_begin:i_end] = lens_pred.numpy().astype(np.float32)
                    g["lens_pred2"][i_begin:i_end] = lens_pred2.numpy().astype(np.float32)
                    g["source_pred"][i_begin:i_end] = tf.transpose(source_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                    g["source_pred2"][i_begin:i_end] = tf.transpose(source_pred2, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                    g["kappa_pred"][i_begin:i_end] = tf.transpose(kappa_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                    g["chi_squared"][i_begin:i_end] = tf.transpose(chi_squared).numpy().astype(np.float32)
                    g["chi_squared2"][i_begin:i_end] = tf.transpose(chi_squared2).numpy().astype(np.float32)
                    g["lens_coherence_spectrum"][i_begin:i_end] = _ps_lens
                    g["lens_coherence_spectrum2"][i_begin:i_end] = _ps_lens2
                    g["source_coherence_spectrum"][i_begin:i_end] = _ps_source
                    g["source_coherence_spectrum2"][i_begin:i_end] = _ps_source2
                    g["lens_coherence_spectrum"][i_begin:i_end] = _ps_lens
                    g["lens_coherence_spectrum"][i_begin:i_end] = _ps_lens
                    g["kappa_coherence_spectrum"][i_begin:i_end] = _ps_kappa

                    if batch == 0:
                        _, f = np.histogram(np.fft.fftfreq(phys.pixels)[:phys.pixels//2], bins=ps_lens.bins)
                        f = (f[:-1] + f[1:]) / 2
                        g["lens_frequencies"][:] = f
                        _, f = np.histogram(np.fft.fftfreq(phys.src_pixels)[:phys.src_pixels//2], bins=ps_source.bins)
                        f = (f[:-1] + f[1:]) / 2
                        g["source_frequencies"][:] = f
                        _, f = np.histogram(np.fft.fftfreq(phys.kappa_pixels)[:phys.kappa_pixels//2], bins=ps_kappa.bins)
                        f = (f[:-1] + f[1:]) / 2
                        g["kappa_frequencies"][:] = f
                        g["kappa_fov"][0] = phys.kappa_fov
                        g["source_fov"][0] = phys.src_fov


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_prefix",       required=True)
    parser.add_argument("--source_model",       required=True,      help="Give the path to a source RIM to converge even further the source")
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--val_dataset",        required=True,      help="Name of the dataset, not full path")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--train_dataset",      required=True)
    parser.add_argument("--train_size",         default=1000,       type=int)
    parser.add_argument("--val_size",           default=10000,      type=int)
    parser.add_argument("--test_size",          default=10000,      type=int)
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--batch_size",         default=1,          type=int)
    parser.add_argument("--lens_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)