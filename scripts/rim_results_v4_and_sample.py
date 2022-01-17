from censai import RIMSharedUnetv3, PhysicalModelv2, PowerSpectrum
import tensorflow_probability as tfp
from censai.models import SharedUnetModelv4
from censai.data.lenses_tng_v3 import decode_results, decode_physical_model_info
import tensorflow as tf
import numpy as np
import os, glob, json
import h5py
from tqdm import tqdm
from censai.definitions import LOG10, DTYPE

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!

SLGD = tfp.optimizer.StochasticGradientLangevinDynamics


def distributed_strategy(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    model = os.path.join(os.getenv('CENSAI_PATH'), "models", args.model)
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

    ps_lens = PowerSpectrum(bins=args.lens_coherence_bins, pixels=physical_params["pixels"].numpy())
    ps_source = PowerSpectrum(bins=args.source_coherence_bins,  pixels=physical_params["src pixels"].numpy())
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

    with open(os.path.join(model, "unet_hparams.json")) as f:
        unet_params = json.load(f)
    unet_params["kernel_l2_amp"] = args.l2_amp
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
    STEPS = args.burn_in + args.sampling_steps

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.experiment_name + "_" + model_name + f"_{THIS_WORKER:02d}.h5"), 'w') as hf:
        for i, dataset in enumerate([train_dataset, val_dataset, test_dataset]):
            g = hf.create_group(f'{dataset_names[i]}')
            data_len = dataset_shapes[i] // N_WORKERS
            g.create_dataset(name="lens", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="psf",  shape=[data_len, physical_params['psf pixels'], physical_params['psf pixels'], 1], dtype=np.float32)
            g.create_dataset(name="psf_fwhm", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="noise_rms", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="source", shape=[data_len, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
            g.create_dataset(name="kappa", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="lens_pred", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="lens_pred_reoptimized", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="lens_pred_reoptimized_mean", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="lens_pred_reoptimized_var",  shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="source_pred", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
            g.create_dataset(name="source_pred_reoptimized_mean", shape=[data_len, phys.src_pixels, phys.src_pixels, 1])
            g.create_dataset(name="source_pred_reoptimized_var", shape=[data_len, phys.src_pixels, phys.src_pixels, 1])
            g.create_dataset(name="kappa_pred", shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="kappa_pred_reoptimized_mean", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="kappa_pred_reoptimized_var", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="chi_squared", shape=[data_len, rim.steps], dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized_mean", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized_series", shape=[data_len, STEPS], dtype=np.float32)
            g.create_dataset(name="source_optim_mse", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="source_optim_mse_series", shape=[data_len, STEPS], dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse_series", shape=[data_len, STEPS], dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum_reoptimized", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum_reoptimized",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
            g.create_dataset(name="kappa_coherence_spectrum", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
            g.create_dataset(name="kappa_coherence_spectrum_reoptimized", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
            g.create_dataset(name="lens_frequencies", shape=[args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="source_frequencies", shape=[args.source_coherence_bins], dtype=np.float32)
            g.create_dataset(name="kappa_frequencies", shape=[args.kappa_coherence_bins], dtype=np.float32)
            g.create_dataset(name="kappa_fov", shape=[1], dtype=np.float32)
            g.create_dataset(name="source_fov", shape=[1], dtype=np.float32)
            g.create_dataset(name="lens_fov", shape=[1], dtype=np.float32)
            dataset = dataset.skip(data_len * (THIS_WORKER - 1)).take(data_len)
            for batch, (lens, source, kappa, noise_rms, psf, fwhm) in enumerate(dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)):
                checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()  # reset model weights
                # Compute predictions for kappa and source
                source_pred, kappa_pred, chi_squared = rim.predict(lens, noise_rms, psf)
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)

                # Re-optimize weights of the model
                learning_rate_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
                    initial_learning_rate=args.learning_rate,
                    decay_rate=args.decay_rate,
                    decay_steps=args.decay_steps,
                    staircase=args.staircase
                )
                # Goodness of fit and oracle statistics
                chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
                source_mse = tf.TensorArray(DTYPE, size=STEPS)
                kappa_mse = tf.TensorArray(DTYPE, size=STEPS)
                y_mean = lens_pred
                y_var = tf.zeros_like(y_mean)
                # prediction and uncertainty collected in model space
                source_mean = source_pred[-1]
                kappa_mean = rim.kappa_inverse_link(kappa_pred)[-1]
                source_var = tf.zeros_like(source_mean)
                kappa_var = tf.zeros_like(kappa_mean)
                # Preconditioner
                grad_var = [tf.zeros_like(theta) for theta in unet.trainable_variables]
                for current_step in tqdm(range(STEPS)):
                    with tf.GradientTape() as tape:
                        tape.watch(unet.trainable_variables)
                        s, k, chi_sq = rim.call(lens, noise_rms, psf, outer_tape=tape)
                        cost = tf.reduce_mean(chi_sq)
                        cost += tf.reduce_sum(rim.unet.losses)  # log prior over the weights

                    log_likelihood = chi_sq[-1]
                    chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood)
                    source_o = s[-1]
                    kappa_o = k[-1]
                    y_o = phys.forward(rim.source_link(source_o), rim.kappa_link(kappa_o), psf)
                    source_mse = source_mse.write(index=current_step, value=tf.reduce_mean((source_o - rim.source_inverse_link(source)) ** 2))
                    kappa_mse = kappa_mse.write(index=current_step, value=tf.reduce_mean((kappa_o - rim.kappa_inverse_link(kappa)) ** 2))
                    # Welford's online algorithm for moving variance
                    if current_step >= args.burn_in:
                        step = current_step - args.burn_in
                        # source
                        delta = source_o - source_mean
                        source_mean = (step * source_mean + source_o) / (step + 1)
                        delta2 = source_o - source_mean
                        source_var += delta * delta2
                        # kappa
                        delta = kappa_o - kappa_mean
                        kappa_mean = (step * kappa_mean + kappa_o) / (step + 1)
                        delta2 = kappa_o - kappa_mean
                        kappa_var += delta * delta2
                        # observation
                        delta = y_o - y_mean
                        y_mean = (step * y_mean + y_o) / (step + 1)
                        delta2 = y_o - y_mean
                        y_var += delta * delta2

                    # Stochastic Gradient Langevin Dynamics (SGLD) update
                    step_size = learning_rate_schedule(current_step)
                    gradients = tape.gradient(cost, unet.trainable_variables)
                    grad_var = [args.rmsprop_alpha * var + (1 - args.rmsprop_alpha) * grad**2 for (grad, var) in zip(gradients, grad_var)]
                    if current_step >= args.slgd_burn_in:
                        noise = [tf.random.normal(shape=theta.shape, stddev=tf.sqrt(step_size / (args.rmsprop_epsilon + tf.sqrt(var)))) for (theta, var) in zip(unet.trainable_variables, grad_var)]
                        unet.set_weights([theta - step_size/2./(args.rmsprop_epsilon + tf.sqrt(var)) * grad + eta
                                          for (theta, grad, eta, var) in zip(unet.trainable_variables, gradients, noise, grad_var)])
                    else:
                        unet.set_weights([theta - step_size/2./(args.rmsprop_epsilon + tf.sqrt(var)) * grad
                                          for (theta, grad, var) in zip(unet.trainable_variables, gradients, grad_var)])

                y_pred = phys.forward(rim.source_link(source_mean), rim.kappa_link(kappa_mean), psf)
                chisq_ro = tf.reduce_mean((y_pred - lens)**2 / noise_rms[:, None, None, None] ** 2)
                chi_sq_mean = tf.reduce_mean((y_mean - lens)**2 / noise_rms[:, None, None, None] ** 2)
                source_mse_ro = tf.reduce_mean((source_mean - rim.source_inverse_link(source)) ** 2)
                kappa_mse_ro = tf.reduce_mean((kappa_mean - rim.kappa_inverse_link(kappa)) ** 2)
                chi_sq_series = tf.transpose(chi_squared_series.stack(), perm=[1, 0])
                source_mse = source_mse.stack()[None, ...]
                kappa_mse = kappa_mse.stack()[None, ...]
                kappa_var /= float(args.sampling_steps)
                source_var /= float(args.sampling_steps)
                y_var /= float(args.sampling_steps)

                # Compute Power spectrum of converged predictions
                _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
                _ps_lens3 = ps_lens.cross_correlation_coefficient(lens[..., 0], y_pred[..., 0])
                _ps_kappa = ps_kappa.cross_correlation_coefficient(rim.kappa_inverse_link(kappa)[..., 0], rim.kappa_inverse_link(kappa_pred[-1])[..., 0])
                _ps_kappa2 = ps_kappa.cross_correlation_coefficient(rim.kappa_inverse_link(kappa)[..., 0], kappa_mean[..., 0])
                _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
                _ps_source3 = ps_source.cross_correlation_coefficient(source[..., 0], source_mean[..., 0])

                # Move mean and variance back to physical space. Assumes link function 10^x for kappa and identity for source.
                kappa_mean = 10**kappa_mean
                kappa_var = kappa_mean**2 * LOG10**2 * kappa_var

                # save results
                g["lens"][batch] = lens.numpy().astype(np.float32)
                g["psf"][batch] = psf.numpy().astype(np.float32)
                g["psf_fwhm"][batch] = fwhm.numpy().astype(np.float32)
                g["noise_rms"][batch] = noise_rms.numpy().astype(np.float32)
                g["source"][batch] = source.numpy().astype(np.float32)
                g["kappa"][batch] = kappa.numpy().astype(np.float32)
                g["lens_pred"][batch] = lens_pred.numpy().astype(np.float32)
                g["lens_pred_reoptimized"][batch] = y_pred.numpy().astype(np.float32)
                g["lens_pred_reoptimized_mean"][batch] = y_mean.numpy().astype(np.float32)
                g["lens_pred_reoptimized_var"][batch] = y_var.numpy().astype(np.float32)
                g["source_pred"][batch] = tf.transpose(source_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["source_pred_reoptimized_mean"][batch] = source_mean.numpy().astype(np.float32)
                g["source_pred_reoptimized_var"][batch] = source_var.numpy().astype(np.float32)
                g["kappa_pred"][batch] = tf.transpose(kappa_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["kappa_pred_reoptimized_mean"][batch] = kappa_mean.numpy().astype(np.float32)
                g["kappa_pred_reoptimized_var"][batch] = kappa_var.numpy().astype(np.float32)
                g["chi_squared"][batch] = 2*tf.transpose(chi_squared).numpy().astype(np.float32)
                g["chi_squared_reoptimized"][batch] = chisq_ro.numpy().astype(np.float32)
                g["chi_squared_reoptimized_mean"][batch] = chi_sq_mean.numpy().astype(np.float32)
                g["chi_squared_reoptimized_series"][batch] = 2*chi_sq_series.numpy().astype(np.float32)
                g["source_optim_mse"][batch] = source_mse_ro.numpy().astype(np.float32)
                g["source_optim_mse_series"][batch] = source_mse.numpy().astype(np.float32)
                g["kappa_optim_mse"][batch] = kappa_mse_ro.numpy().astype(np.float32)
                g["kappa_optim_mse_series"][batch] = kappa_mse.numpy().astype(np.float32)
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["lens_coherence_spectrum_reoptimized"][batch] = _ps_lens3
                g["source_coherence_spectrum"][batch] = _ps_source
                g["source_coherence_spectrum_reoptimized"][batch] = _ps_source3
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["kappa_coherence_spectrum"][batch] = _ps_kappa
                g["kappa_coherence_spectrum_reoptimized"][batch] = _ps_kappa2

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
    parser.add_argument("--experiment_name",    default="reoptim")
    parser.add_argument("--model",              required=True,      help="Model to get predictions from")
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--val_dataset",        required=True,      help="Name of the dataset, not full path")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--train_dataset",      required=True)
    parser.add_argument("--train_size",         default=0,        type=int)
    parser.add_argument("--val_size",           default=0,        type=int)
    parser.add_argument("--test_size",          default=5000,       type=int)
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--lens_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)
    parser.add_argument("--burn_in",            default=2000,       type=int)
    parser.add_argument("--slgd_burn_in",       default=2000,       type=int, help="Burn in before injecting noise to the gradient to collect preconditionner statistics")
    parser.add_argument("--sampling_steps",     default=2000,       type=int)
    parser.add_argument("--l2_amp",             default=6e-5,       type=float)
    parser.add_argument("--rmsprop_alpha",      default=0.99,       type=float, help="Control the size of the exponential moving window of the preconditioner. Default from Li et al 2015.")
    parser.add_argument("--rmsprop_epsilon",    default=1e-5,       type=float, help="Control the extremes of the curvatures in the preconditioner. Default from Li et al 2015.")
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=1,          type=float)
    parser.add_argument("--decay_steps",        default=500,        type=float)
    parser.add_argument("--staircase",          action="store_true")
    parser.add_argument("--seed",               required=True,      type=int,   help="Seed for shuffling to make sure all job agree on dataset order.")

    args = parser.parse_args()
    distributed_strategy(args)