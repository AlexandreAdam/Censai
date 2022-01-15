from censai import RIMSharedUnetv3, PhysicalModelv2, PowerSpectrum
from censai.models import SharedUnetModelv4, VAE
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
    files = glob.glob(os.path.join(os.getenv('CENSAI_PATH'), "data", args.test_dataset, "*.tfrecords"))
    files = tf.data.Dataset.from_tensor_slices(files)
    test_dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type=args.compression_type).shuffle(len(files)), block_length=1, num_parallel_calls=tf.data.AUTOTUNE)
    # Read off global parameters from first example in dataset
    for physical_params in test_dataset.map(decode_physical_model_info):
        break
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
    rim_params["source_link"] = "relu"
    rim = RIMSharedUnetv3(phys, unet, **rim_params)

    kvae_path = os.path.join(os.getenv('CENSAI_PATH'), "models", args.kappa_vae)
    with open(os.path.join(kvae_path, "model_hparams.json"), "r") as f:
        kappa_vae_hparams = json.load(f)
    kappa_vae = VAE(**kappa_vae_hparams)
    ckpt1 = tf.train.Checkpoint(step=tf.Variable(1), net=kappa_vae)
    checkpoint_manager1 = tf.train.CheckpointManager(ckpt1, kvae_path, 1)
    checkpoint_manager1.checkpoint.restore(checkpoint_manager1.latest_checkpoint).expect_partial()

    svae_path = os.path.join(os.getenv('CENSAI_PATH'), "models", args.source_vae)
    with open(os.path.join(svae_path, "model_hparams.json"), "r") as f:
        source_vae_hparams = json.load(f)
    source_vae = VAE(**source_vae_hparams)
    ckpt2 = tf.train.Checkpoint(step=tf.Variable(1), net=source_vae)
    checkpoint_manager2 = tf.train.CheckpointManager(ckpt2, svae_path, 1)
    checkpoint_manager2.checkpoint.restore(checkpoint_manager2.latest_checkpoint).expect_partial()

    dataset_names = [args.test_dataset]
    dataset_shapes = [args.test_size]
    model_name = os.path.split(model)[-1]
    wk = tf.keras.layers.Lambda(lambda k: tf.sqrt(k) / tf.reduce_sum(tf.sqrt(k), axis=(1, 2, 3), keepdims=True))
    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.experiment_name + "_" + model_name + f"_{THIS_WORKER:02d}.h5"), 'w') as hf:
        for i, dataset in enumerate([test_dataset]):
            g = hf.create_group(f'{dataset_names[i]}')
            data_len = dataset_shapes[i] // N_WORKERS
            g.create_dataset(name="lens",                                   shape=[data_len, phys.pixels, phys.pixels, 1],                                      dtype=np.float32)
            g.create_dataset(name="psf",                                    shape=[data_len, physical_params['psf pixels'], physical_params['psf pixels'], 1],  dtype=np.float32)
            g.create_dataset(name="psf_fwhm",                               shape=[data_len],                                                                   dtype=np.float32)
            g.create_dataset(name="noise_rms",                              shape=[data_len],                                                                   dtype=np.float32)
            g.create_dataset(name="source",                                 shape=[data_len, phys.src_pixels, phys.src_pixels, 1],                              dtype=np.float32)
            g.create_dataset(name="kappa",                                  shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1],                          dtype=np.float32)
            g.create_dataset(name="lens_prior",                             shape=[data_len, phys.pixels, phys.pixels, 1],                                      dtype=np.float32)
            g.create_dataset(name="lens_pred",                              shape=[data_len, phys.pixels, phys.pixels, 1],                                      dtype=np.float32)
            g.create_dataset(name="lens_pred_reoptimized",                  shape=[data_len, phys.pixels, phys.pixels, 1],                                      dtype=np.float32)
            g.create_dataset(name="source_prior",                           shape=[data_len, phys.src_pixels, phys.src_pixels, 1],                   dtype=np.float32)
            g.create_dataset(name="source_pred",                            shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1],                   dtype=np.float32)
            g.create_dataset(name="source_pred_reoptimized",                shape=[data_len, phys.src_pixels, phys.src_pixels, 1],                              dtype=np.float32)
            g.create_dataset(name="kappa_prior",                            shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1],               dtype=np.float32)
            g.create_dataset(name="kappa_pred",                             shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1],               dtype=np.float32)
            g.create_dataset(name="kappa_pred_reoptimized",                 shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1],                          dtype=np.float32)
            g.create_dataset(name="chi_squared",                            shape=[data_len, rim.steps],                                                        dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized",                shape=[data_len, rim.steps],                                                                   dtype=np.float32)
            g.create_dataset(name="chi_squared_reoptimized_series",         shape=[data_len, rim.steps, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="sampled_chi_squared_reoptimized_series", shape=[data_len, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="source_optim_mse",                       shape=[data_len],                                                                   dtype=np.float32)
            g.create_dataset(name="source_optim_mse_series",                shape=[data_len, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="sampled_source_optim_mse_series",        shape=[data_len, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse",                        shape=[data_len],                                                                   dtype=np.float32)
            g.create_dataset(name="kappa_optim_mse_series",                 shape=[data_len, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="sampled_kappa_optim_mse_series",         shape=[data_len, args.re_optimize_steps],                                           dtype=np.float32)
            g.create_dataset(name="latent_kappa_gt_distance_init",          shape=[data_len, kappa_vae.latent_size],                                            dtype=np.float32)
            g.create_dataset(name="latent_source_gt_distance_init",         shape=[data_len, source_vae.latent_size],                                           dtype=np.float32)
            g.create_dataset(name="latent_kappa_gt_distance_end",           shape=[data_len, kappa_vae.latent_size],                                            dtype=np.float32)
            g.create_dataset(name="latent_source_gt_distance_end",          shape=[data_len, source_vae.latent_size],                                           dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum",                shape=[data_len, args.lens_coherence_bins],                                         dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum",              shape=[data_len, args.source_coherence_bins],                                       dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum2",               shape=[data_len, args.lens_coherence_bins],                                         dtype=np.float32)
            g.create_dataset(name="lens_coherence_spectrum_reoptimized",    shape=[data_len, args.lens_coherence_bins],                                         dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum2",             shape=[data_len, args.source_coherence_bins],                                       dtype=np.float32)
            g.create_dataset(name="source_coherence_spectrum_reoptimized",  shape=[data_len, args.source_coherence_bins],                                       dtype=np.float32)
            g.create_dataset(name="kappa_coherence_spectrum",               shape=[data_len, args.kappa_coherence_bins],                                        dtype=np.float32)
            g.create_dataset(name="kappa_coherence_spectrum_reoptimized",   shape=[data_len, args.kappa_coherence_bins],                                        dtype=np.float32)
            g.create_dataset(name="lens_frequencies",                       shape=[args.lens_coherence_bins],                                                   dtype=np.float32)
            g.create_dataset(name="source_frequencies",                     shape=[args.source_coherence_bins],                                                 dtype=np.float32)
            g.create_dataset(name="kappa_frequencies",                      shape=[args.kappa_coherence_bins],                                                  dtype=np.float32)
            g.create_dataset(name="kappa_fov",                              shape=[1],                                                                          dtype=np.float32)
            g.create_dataset(name="source_fov",                             shape=[1],                                                                          dtype=np.float32)
            g.create_dataset(name="lens_fov",                               shape=[1],                                                                          dtype=np.float32)
            dataset = dataset.skip(data_len * (THIS_WORKER - 1)).take(data_len)
            for batch, (lens, source, kappa, noise_rms, psf, fwhm) in enumerate(dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)):
                checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()  # reset model weights

                # RIM predictions for kappa and source
                source_pred, kappa_pred, chi_squared = rim.predict(lens, noise_rms, psf)
                lens_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)
                source_o = source_pred[-1]
                kappa_o = kappa_pred[-1]

                # ===================== VAE SAMPLING for PRIOR ==============================
                # Ground truth latent code for oracle metrics
                z_source_gt, _ = source_vae.encoder(source)
                z_kappa_gt, _ = kappa_vae.encoder(log_10(kappa))

                # Latent code of model predictions
                z_source, _ = source_vae.encoder(source_o)
                z_kappa, _ = kappa_vae.encoder(log_10(kappa_o))

                # L1 distance with ground truth in latent space -- this is changed by an user defined value when using real data
                # z_source_std = tf.abs(z_source - z_source_gt)
                # z_kappa_std = tf.abs(z_kappa - z_kappa_gt)
                z_source_std = args.source_vae_ball_size
                z_kappa_std = args.kappa_vae_ball_size

                # Sample latent code, then decode and forward
                z_s = tf.random.normal(shape=[args.sample_size, source_vae.latent_size], mean=z_source, stddev=z_source_std)
                z_k = tf.random.normal(shape=[args.sample_size, kappa_vae.latent_size],  mean=z_kappa,  stddev=z_kappa_std)
                sampled_source = tf.nn.relu(source_vae.decode(z_s))
                sampled_source /= tf.reduce_max(sampled_source, axis=(1, 2, 3), keepdims=True)  # make sampled source normalized and positive
                sampled_kappa = kappa_vae.decode(z_k)  # output in log_10 space
                sampled_lens = phys.noisy_forward(sampled_source, 10**sampled_kappa, noise_rms, tf.tile(psf, [args.sample_size, 1, 1, 1]))  # for posterity and fishing experiments

                # Re-optimize weights of the model
                STEPS = args.re_optimize_steps
                learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=args.learning_rate,
                    decay_rate=args.decay_rate,
                    decay_steps=args.decay_steps,
                    staircase=args.staircase
                )
                optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)

                chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
                source_mse = tf.TensorArray(DTYPE, size=STEPS)
                kappa_mse = tf.TensorArray(DTYPE, size=STEPS)
                # sampled_chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
                sampled_source_mse = tf.TensorArray(DTYPE, size=STEPS)
                sampled_kappa_mse = tf.TensorArray(DTYPE, size=STEPS)

                best = chi_squared
                source_best = source_pred[-1]
                kappa_best = kappa_pred[-1]
                for current_step in tqdm(range(STEPS)):
                    with tf.GradientTape() as tape:
                        tape.watch(unet.trainable_variables)
                        # log likelihood
                        # s, k, chi_sq = rim.call(lens, noise_rms, psf, outer_tape=tape) # was using this in experiment 1 and 2 + only single sample for mse
                        # cost = tf.reduce_mean(chi_sq)
                        # log prior
                        s, k, chi_sq = rim.call(sampled_lens, noise_rms, psf, outer_tape=tape)
                        _kappa_mse = tf.reduce_sum(wk(10**sampled_kappa) * (k - sampled_kappa) ** 2, axis=(2, 3, 4))
                        cost = tf.reduce_mean(_kappa_mse)
                        cost += tf.reduce_mean((s - sampled_source)**2)
                        cost += tf.reduce_sum(rim.unet.losses)  # weight decay

                    grads = tape.gradient(cost, unet.trainable_variables)
                    optim.apply_gradients(zip(grads, unet.trainable_variables))

                    # Record performance on sampled dataset
                    sampled_chi_squared_series = sampled_chi_squared_series.write(index=current_step, value=tf.squeeze(tf.reduce_mean(chi_sq[-1])))
                    sampled_source_mse = sampled_source_mse.write(index=current_step, value=tf.reduce_mean((s[-1] - sampled_source) ** 2))
                    sampled_kappa_mse = sampled_kappa_mse.write(index=current_step, value=tf.reduce_mean((k[-1] - sampled_kappa) ** 2))
                    # Record model prediction on data
                    s, k, chi_sq = rim.call(lens, noise_rms, psf, outer_tape=tape)
                    chi_squared_series = chi_squared_series.write(index=current_step, value=tf.squeeze(chi_sq))
                    source_o = s[-1]
                    kappa_o = k[-1]
                    # oracle metrics, remove when using real data
                    source_mse = source_mse.write(index=current_step, value=tf.reduce_mean((source_o - source) ** 2))
                    kappa_mse = kappa_mse.write(index=current_step, value=tf.reduce_mean((kappa_o - log_10(kappa)) ** 2))
                    # Early stopping could be replaced altogether by window averaging... (could we use SLGD here? In the mean time use approximate sampling from RMSprop)
                    # if 2 * chi_sq[-1, 0] < args.converged_chisq:
                    #     source_best = source_o
                    #     kappa_best = 10**kappa_o
                    #     best = chi_sq[-1, 0]
                    #     break
                    if chi_sq[-1, 0] < best[-1, 0]:
                        source_best = tf.nn.relu(source_o)
                        kappa_best = 10**kappa_o
                        best = chi_sq
                        source_mse_best = tf.reduce_mean((source_best - source) ** 2)
                        kappa_mse_best = tf.reduce_mean((kappa_best - log_10(kappa)) ** 2)

                source_o = source_best
                kappa_o = kappa_best
                y_pred = phys.forward(source_o, kappa_o, psf)
                chi_sq_series = tf.transpose(chi_squared_series.stack())
                source_mse = source_mse.stack()
                kappa_mse = kappa_mse.stack()
                sampled_chi_squared_series = sampled_chi_squared_series.stack()
                sampled_source_mse = sampled_source_mse.stack()
                sampled_kappa_mse = sampled_kappa_mse.stack()

                # Latent code of optimized model predictions
                z_source_opt, _ = source_vae.encoder(tf.nn.relu(source_o))
                z_kappa_opt, _ = kappa_vae.encoder(log_10(kappa_o))

                # Compute Power spectrum of converged predictions
                _ps_lens = ps_lens.cross_correlation_coefficient(lens[..., 0], lens_pred[..., 0])
                _ps_lens3 = ps_lens.cross_correlation_coefficient(lens[..., 0], y_pred[..., 0])
                _ps_kappa = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
                _ps_kappa2 = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_o[..., 0]))
                _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
                _ps_source3 = ps_source.cross_correlation_coefficient(source[..., 0], source_o[..., 0])

                # save results
                g["lens"][batch] = lens.numpy().astype(np.float32)
                g["psf"][batch] = psf.numpy().astype(np.float32)
                g["psf_fwhm"][batch] = fwhm.numpy().astype(np.float32)
                g["noise_rms"][batch] = noise_rms.numpy().astype(np.float32)
                g["source"][batch] = source.numpy().astype(np.float32)
                g["kappa"][batch] = kappa.numpy().astype(np.float32)
                g["lens_prior"][batch] = sampled_lens[0].numpy().astype(np.float32)
                g["lens_pred"][batch] = lens_pred.numpy().astype(np.float32)
                g["lens_pred_reoptimized"][batch] = y_pred.numpy().astype(np.float32)
                g["source_prior"][batch] = sampled_source[0].numpy().astype(np.float32)
                g["source_pred"][batch] = tf.transpose(source_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["source_pred_reoptimized"][batch] = source_o.numpy().astype(np.float32)
                g["kappa_prior"][batch] = 10**sampled_kappa[0].numpy().astype(np.float32)
                g["kappa_pred"][batch] = tf.transpose(kappa_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
                g["kappa_pred_reoptimized"][batch] = kappa_o.numpy().astype(np.float32)
                g["chi_squared"][batch] = 2*tf.squeeze(chi_squared).numpy().astype(np.float32)
                g["chi_squared_reoptimized"][batch] = 2*tf.squeeze(best).numpy().astype(np.float32)
                g["chi_squared_reoptimized_series"][batch] = 2*chi_sq_series.numpy().astype(np.float32)
                g["sampled_chi_squared_reoptimized_series"][batch] = 2*sampled_chi_squared_series.numpy().astype(np.float32)
                g["source_optim_mse"][batch] = source_mse_best.numpy().astype(np.float32)
                g["source_optim_mse_series"][batch] = source_mse.numpy().astype(np.float32)
                g["sampled_source_optim_mse_series"][batch] = sampled_source_mse.numpy().astype(np.float32)
                g["kappa_optim_mse"][batch] = kappa_mse_best.numpy().astype(np.float32)
                g["kappa_optim_mse_series"][batch] = kappa_mse.numpy().astype(np.float32)
                g["sampled_kappa_optim_mse_series"][batch] = sampled_kappa_mse.numpy().astype(np.float32)
                g["latent_source_gt_distance_init"][batch] = tf.abs(z_source - z_source_gt).numpy().squeeze().astype(np.float32)
                g["latent_kappa_gt_distance_init"][batch] = tf.abs(z_kappa - z_kappa_gt).numpy().squeeze().astype(np.float32)
                g["latent_source_gt_distance_end"][batch] = tf.abs(z_source_opt - z_source_gt).numpy().squeeze().astype(np.float32)
                g["latent_kappa_gt_distance_end"][batch] = tf.abs(z_kappa_opt - z_kappa_gt).numpy().squeeze().astype(np.float32)
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
    parser.add_argument("--source_vae",         required=True)
    parser.add_argument("--kappa_vae",          required=True)
    parser.add_argument("--compression_type",   default="GZIP")
    parser.add_argument("--test_dataset",       required=True)
    parser.add_argument("--test_size",          default=5000,       type=int)
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--sample_size",        default=10,         type=int)
    parser.add_argument("--lens_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)
    parser.add_argument("--re_optimize_steps",  default=2000,       type=int)
    parser.add_argument("--converged_chisq",    default=1.0,        type=float, help="Value of the chisq that is considered converged.")
    parser.add_argument("--source_vae_ball_size",   default=0.5,    type=float, help="Standard deviation of the source VAE latent space sampling around RIM prediction")
    parser.add_argument("--kappa_vae_ball_size",    default=0.5,    type=float, help="Standard deviation of the kappa VAE latent space sampling around RIM prediction")
    parser.add_argument("--l2_amp",             default=1e-6,       type=float)
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=1,          type=float)
    parser.add_argument("--decay_steps",        default=50,         type=float)
    parser.add_argument("--staircase",          action="store_true")
    parser.add_argument("--seed",               required=True,      type=int,   help="Seed for shuffling to make sure all job agree on dataset order.")

    args = parser.parse_args()
    distributed_strategy(args)