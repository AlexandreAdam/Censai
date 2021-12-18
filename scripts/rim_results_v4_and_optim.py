from censai import RIMSharedUnetv3, PhysicalModelv2, PowerSpectrum, RIMSourceUnetv2, AnalyticalPhysicalModel
from censai.models import SharedUnetModelv4, UnetModelv2
from censai.data.lenses_tng_v3 import decode_results, decode_physical_model_info
import tensorflow as tf
import numpy as np
import os, glob, json
import h5py
from tqdm import tqdm
from censai.definitions import log_10, DTYPE

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


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
    #
    # from censai.utils import nulltape
    # def call_with_mask(self, lensed_image, noise_rms, psf, mask, outer_tape=nulltape):
    #     """
    #     Used in training. Return linked kappa and source maps.
    #     """
    #     batch_size = lensed_image.shape[0]
    #     source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
    #     source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad,
    #                                            states)  # Use lens to make an initial guess with Unet
    #     source_series = tf.TensorArray(DTYPE, size=self.steps)
    #     kappa_series = tf.TensorArray(DTYPE, size=self.steps)
    #     chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
    #     # record initial guess
    #     source_series = source_series.write(index=0, value=source)
    #     kappa_series = kappa_series.write(index=0, value=kappa)
    #     # Main optimization loop
    #     for current_step in tf.range(self.steps - 1):
    #         with outer_tape.stop_recording():
    #             with tf.GradientTape() as g:
    #                 g.watch(source)
    #                 g.watch(kappa)
    #                 y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
    #                 flux_term = tf.square(
    #                     tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
    #                 log_likelihood = 0.5 * tf.reduce_sum(
    #                     tf.square(y_pred - mask * lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
    #                 cost = tf.reduce_mean(log_likelihood + self.flux_lagrange_multiplier * flux_term)
    #             source_grad, kappa_grad = g.gradient(cost, [source, kappa])
    #             source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
    #         source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)
    #         source_series = source_series.write(index=current_step + 1, value=source)
    #         kappa_series = kappa_series.write(index=current_step + 1, value=kappa)
    #         chi_squared_series = chi_squared_series.write(index=current_step,
    #                                                       value=log_likelihood / self.pixels ** 2)  # renormalize chi squared here
    #     # last step score
    #     log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source),
    #                                                         kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
    #     chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
    #     return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", model_name + "_" + args.source_model + f"_{THIS_WORKER:02d}.h5"), 'w') as hf:
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
            g.create_dataset(name="lens_pred2", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="lens_pred_reoptimized", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
            g.create_dataset(name="source_pred", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
            g.create_dataset(name="source_pred2", shape=[data_len,  rim_source.steps,  phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
            g.create_dataset(name="source_pred_reoptimized", shape=[data_len, phys.src_pixels, phys.src_pixels, 1])
            g.create_dataset(name="kappa_pred", shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="kappa_pred_reoptimized", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
            g.create_dataset(name="chi_squared", shape=[data_len, rim.steps], dtype=np.float32)
            g.create_dataset(name="chi_squared2", shape=[data_len, rim_source.steps], dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
            g.create_dataset(name="source_optim_mse", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
            g.create_dataset(name="source_optim_mse", shape=[data_len], dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum2", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum_repotimized", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum2",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
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
                # Re-optimize source with a trained source model
                source_pred2, chi_squared2 = rim_source.predict(lens, kappa_pred[-1], noise_rms, psf)
                lens_pred2 = phys.forward(source_pred2[-1], kappa_pred[-1], psf)
                # Re-optimize weights of the model
                STEPS = args.re_optimize_steps
                learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=args.learning_rate,
                    decay_rate=args.decay_rate,
                    decay_steps=args.decay_steps,
                    staircase=args.staircase
                )
                optim = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

                chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
                source_mse = tf.TensorArray(DTYPE, size=STEPS)
                kappa_mse = tf.TensorArray(DTYPE, size=STEPS)
                best = chi_squared[-1, 0]
                source_best = source_pred[-1]
                kappa_best = kappa_pred[-1]
                for current_step in tqdm(range(STEPS)):
                    with tf.GradientTape() as tape:
                        tape.watch(unet.trainable_variables)
                        # s, k, chi_sq = call_with_mask(rim, lens, noise_rms, psf, mask, tape)
                        s, k, chi_sq = rim.call(lens, noise_rms, psf, outer_tape=tape)
                        cost = tf.reduce_mean(chi_sq) # mean over time steps
                        cost += tf.reduce_sum(rim.unet.losses)

                    log_likelihood = chi_sq[-1]
                    chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood)
                    source_o = s[-1]
                    kappa_o = k[-1]
                    source_mse = source_mse.write(index=current_step, value=tf.reduce_mean((source_o - rim.source_inverse_link(source)) ** 2))
                    kappa_mse = kappa_mse.write(index=current_step, value=tf.reduce_mean((kappa_o - rim.kappa_inverse_link(kappa)) ** 2))
                    if 2 * chi_sq[-1, 0] < args.converged_chisq:
                        source_best = rim.source_link(source_o)
                        kappa_best = rim.kappa_link(kappa_o)
                        best = chi_sq[-1, 0]
                        break
                    if chi_sq[-1, 0] < best:
                        source_best = rim.source_link(source_o)
                        kappa_best = rim.kappa_link(kappa_o)
                        best = chi_sq[-1, 0]
                        source_mse_best = tf.reduce_mean((source_best - rim.source_inverse_link(source)) ** 2)
                        kappa_mse_best = tf.reduce_mean((kappa_best - rim.kappa_inverse_link(kappa)) ** 2)
                    grads = tape.gradient(cost, unet.trainable_variables)
                    optim.apply_gradients(zip(grads, unet.trainable_variables))

                source_o = source_best
                kappa_o = kappa_best
                y_pred = phys.forward(source_o, kappa_o, psf)
                chi_sq_series = tf.transpose(chi_squared_series.stack(), perm=[1, 0])
                source_mse = source_mse.stack()[None, ...]
                kappa_mse = kappa_mse.stack()[None, ...]

                # Compute Power spectrum of converged predictions
                _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
                _ps_lens2 = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred2[..., 0])
                _ps_lens3 = ps_lens.cross_correlation_coefficient(lens[..., 0], y_pred[..., 0])
                _ps_kappa = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
                _ps_kappa2 = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_o[..., 0]))
                _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
                _ps_source2 = ps_source.cross_correlation_coefficient(source[..., 0], source_pred2[-1][..., 0])
                _ps_source3 = ps_source.cross_correlation_coefficient(source[..., 0], source_o[..., 0])

                # save results
                g["lens"][batch] = lens.numpy().astype(np.float32)
                g["psf"][batch] = psf.numpy().astype(np.float32)
                g["psf_fwhm"][batch] = fwhm.numpy().astype(np.float32)
                g["noise_rms"][batch] = noise_rms.numpy().astype(np.float32)
                g["source"][batch] = source.numpy().astype(np.float32)
                g["kappa"][batch] = kappa.numpy().astype(np.float32)
                g["lens_pred"][batch] = lens_pred.numpy().astype(np.float32)
                g["lens_pred2"][batch] = lens_pred2.numpy().astype(np.float32)
                g["lens_pred_reoptimized"][batch] = y_pred.numpy().astype(np.float32)
                g["source_pred"][batch] = tf.transpose(source_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["source_pred2"][batch] = tf.transpose(source_pred2, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["source_pred_reoptimized"][batch] = source_o.numpy().astype(np.float32)
                g["kappa_pred"][batch] = tf.transpose(kappa_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["kappa_pred_reoptimized"][batch] = kappa_o.numpy().astype(np.float32)
                g["chi_squared"][batch] = 2*tf.transpose(chi_squared).numpy().astype(np.float32)
                g["chi_squared2"][batch] = 2*tf.transpose(chi_squared2).numpy().astype(np.float32)
                g["chi_squared_reoptimized"][batch] = 2*best.numpy().astype(np.float32)
                g["chi_squared_reoptimized_series"][batch] = 2*chi_sq_series.numpy().astype(np.float32)
                g["source_optim_mse"][batch] = source_mse_best.numpy().astype(np.float32)
                g["source_optim_mse_series"][batch] = source_mse.numpy().astype(np.float32)
                g["kappa_optim_mse"][batch] = kappa_mse_best.numpy().astype(np.float32)
                g["kappa_optim_mse_series"][batch] = kappa_mse.numpy().astype(np.float32)
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["lens_coherence_spectrum2"][batch] = _ps_lens2
                g["source_coherence_spectrum"][batch] = _ps_source
                g["source_coherence_spectrum2"][batch] = _ps_source2
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["lens_coherence_spectrum"][batch] = _ps_lens
                g["kappa_coherence_spectrum"][batch] = _ps_kappa

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

        # Create SIE test
        g = hf.create_group(f'SIE_test')
        data_len = args.sie_size // N_WORKERS
        sie_dataset = test_dataset.skip(data_len * (THIS_WORKER - 1)).take(data_len)
        g.create_dataset(name="lens", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        g.create_dataset(name="psf",  shape=[data_len, physical_params['psf pixels'], physical_params['psf pixels'], 1], dtype=np.float32)
        g.create_dataset(name="psf_fwhm", shape=[data_len], dtype=np.float32)
        g.create_dataset(name="noise_rms", shape=[data_len], dtype=np.float32)
        g.create_dataset(name="source", shape=[data_len, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        g.create_dataset(name="kappa", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        g.create_dataset(name="lens_pred", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        g.create_dataset(name="lens_pred2", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        g.create_dataset(name="source_pred", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        g.create_dataset(name="source_pred2", shape=[data_len, rim_source.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        g.create_dataset(name="kappa_pred", shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        g.create_dataset(name="chi_squared", shape=[data_len, rim.steps], dtype=np.float32)
        g.create_dataset(name="chi_squared2", shape=[data_len, rim_source.steps], dtype=np.float32)
        g.create_dataset(name="lens_coherence_spectrum", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
        g.create_dataset(name="source_coherence_spectrum",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
        g.create_dataset(name="lens_coherence_spectrum2", shape=[data_len, args.lens_coherence_bins], dtype=np.float32)
        g.create_dataset(name="source_coherence_spectrum2",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
        g.create_dataset(name="kappa_coherence_spectrum", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
        g.create_dataset(name="lens_frequencies", shape=[args.lens_coherence_bins], dtype=np.float32)
        g.create_dataset(name="source_frequencies", shape=[args.source_coherence_bins], dtype=np.float32)
        g.create_dataset(name="kappa_frequencies", shape=[args.kappa_coherence_bins], dtype=np.float32)
        g.create_dataset(name="einstein_radius", shape=[data_len], dtype=np.float32)
        g.create_dataset(name="position", shape=[data_len, 2], dtype=np.float32)
        g.create_dataset(name="orientation", shape=[data_len], dtype=np.float32)
        g.create_dataset(name="ellipticity", shape=[data_len], dtype=np.float32)
        g.create_dataset(name="kappa_fov", shape=[1], dtype=np.float32)
        g.create_dataset(name="source_fov", shape=[1], dtype=np.float32)
        g.create_dataset(name="lens_fov", shape=[1], dtype=np.float32)

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
            # Compute Power spectrum of converged predictions
            _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
            _ps_lens2 = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred2[..., 0])
            _ps_kappa = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
            _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
            _ps_source2 = ps_source.cross_correlation_coefficient(source[..., 0], source_pred2[-1][..., 0])

            # save results
            i_begin = batch * args.batch_size
            i_end = i_begin + batch_size
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
            g["chi_squared"][i_begin:i_end] = 2*tf.transpose(chi_squared).numpy().astype(np.float32)
            g["chi_squared2"][i_begin:i_end] = 2*tf.transpose(chi_squared2).numpy().astype(np.float32)
            g["lens_coherence_spectrum"][i_begin:i_end] = _ps_lens.numpy().astype(np.float32)
            g["lens_coherence_spectrum2"][i_begin:i_end] = _ps_lens2.numpy().astype(np.float32)
            g["source_coherence_spectrum"][i_begin:i_end] = _ps_source.numpy().astype(np.float32)
            g["source_coherence_spectrum2"][i_begin:i_end] = _ps_source2.numpy().astype(np.float32)
            g["kappa_coherence_spectrum"][i_begin:i_end] = _ps_kappa.numpy().astype(np.float32)
            g["einstein_radius"][i_begin:i_end] = einstein_radius[:, 0, 0, 0].numpy().astype(np.float32)
            g["position"][i_begin:i_end] = tf.stack([x0[:, 0, 0, 0], y0[:, 0, 0, 0]], axis=1).numpy().astype(np.float32)
            g["ellipticity"][i_begin:i_end] = ellipticity[:, 0, 0, 0].numpy().astype(np.float32)
            g["orientation"][i_begin:i_end] = phi[:, 0, 0, 0].numpy().astype(np.float32)

            if batch == 0:
                _, f = np.histogram(np.fft.fftfreq(phys.pixels)[:phys.pixels // 2], bins=ps_lens.bins)
                f = (f[:-1] + f[1:]) / 2
                g["lens_frequencies"][:] = f
                _, f = np.histogram(np.fft.fftfreq(phys.src_pixels)[:phys.src_pixels // 2], bins=ps_source.bins)
                f = (f[:-1] + f[1:]) / 2
                g["source_frequencies"][:] = f
                _, f = np.histogram(np.fft.fftfreq(phys.kappa_pixels)[:phys.kappa_pixels // 2], bins=ps_kappa.bins)
                f = (f[:-1] + f[1:]) / 2
                g["kappa_frequencies"][:] = f
                g["kappa_fov"][0] = phys.kappa_fov
                g["source_fov"][0] = phys.src_fov


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model",              required=True,      help="Model to get predictions from")
    parser.add_argument("--source_model",       required=True,      help="Give the path to a source RIM to converge even further the source")
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--val_dataset",        required=True,      help="Name of the dataset, not full path")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--train_dataset",      required=True)
    parser.add_argument("--train_size",         default=100,        type=int)
    parser.add_argument("--sie_size",           default=100,        type=int)
    parser.add_argument("--val_size",           default=100,        type=int)
    parser.add_argument("--test_size",          default=5000,       type=int)
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--lens_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)
    parser.add_argument("--batch_size",         default=1,          help="For SIE")
    parser.add_argument("--re_optimize_steps",  default=1000,       type=int)
    # parser.add_argument("--re_optimize_save",   default=4,          type=int)
    parser.add_argument("--converged_chisq",    default=1.0,      type=float, help="Value of the chisq that is considered converged.")
    # parser.add_argument("--convergence_criteria", default=1e-4,     type=float, help="How close should the prediction be to 1?")
    parser.add_argument("--l2_amp",             default=1e-2,       type=float)
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=0.7,        type=float)
    parser.add_argument("--decay_steps",        default=50,         type=float)
    parser.add_argument("--staircase",          action="store_true")
    parser.add_argument("--max_shift",          default=0.1,        type=float, help="Maximum allowed shift of kappa map center in arcseconds")
    parser.add_argument("--max_ellipticity",    default=0.6,        type=float, help="Maximum ellipticty of density profile.")
    parser.add_argument("--max_theta_e",        default=0.5,        type=float, help="Maximum allowed Einstein radius")
    parser.add_argument("--min_theta_e",        default=2.5,        type=float, help="Minimum allowed Einstein radius")
    parser.add_argument("--mask_sigma_threshold",  default=3,          type=float, help="Mask any pixels below threshold * noise_rms for reoptimization")
    parser.add_argument("--seed",               required=True,      type=int,   help="Seed for shuffling to make sure all job agree on dataset order.")

    args = parser.parse_args()
    distributed_strategy(args)