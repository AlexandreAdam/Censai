from censai import RIMSharedUnetv3 as RIM, PhysicalModelv2 as PhysicalModel, PowerSpectrum, EWC
from censai.models import SharedUnetModelv4 as Model, VAE
# from censai.data.lenses_tng_v3 import decode_results, decode_physical_model_info
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

    model = os.path.join(os.getenv('CENSAI_PATH'), "models", args.model)
    path = os.getenv('CENSAI_PATH') + "/results/"
    dataset = []
    for file in sorted(glob.glob(path + args.h5_pattern)):
        try:
            dataset.append(h5py.File(file, "r"))
        except:
            continue
    B = dataset[0]["source"].shape[0]
    data_len = len(dataset) * B // N_WORKERS

    ps_observation = PowerSpectrum(bins=args.observation_coherence_bins, pixels=128)
    ps_source = PowerSpectrum(bins=args.source_coherence_bins,  pixels=128)
    ps_kappa = PowerSpectrum(bins=args.kappa_coherence_bins,  pixels=128)

    phys = PhysicalModel(
        pixels=128,
        kappa_pixels=128,
        src_pixels=128,
        image_fov=7.69,
        kappa_fov=7.69,
        src_fov=3.,
        method="fft",
    )

    with open(os.path.join(model, "unet_hparams.json")) as f:
        unet_params = json.load(f)
    unet_params["kernel_l2_amp"] = args.l2_amp
    unet = Model(**unet_params)
    ckpt = tf.train.Checkpoint(net=unet)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, model, 1)
    checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    with open(os.path.join(model, "rim_hparams.json")) as f:
        rim_params = json.load(f)
    rim = RIM(phys, unet, **rim_params)

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
    wk = lambda k: tf.sqrt(k) / tf.reduce_sum(tf.sqrt(k), axis=(1, 2, 3), keepdims=True)

    # Freeze L5
    # encoding layers
    # rim.unet.layers[0].trainable = False # L1
    # rim.unet.layers[1].trainable = False
    # rim.unet.layers[2].trainable = False
    # rim.unet.layers[3].trainable = False
    rim.unet.layers[4].trainable = False # L5
    # GRU
    # rim.unet.layers[5].trainable = False
    # rim.unet.layers[6].trainable = False
    # rim.unet.layers[7].trainable = False
    # rim.unet.layers[8].trainable = False
    rim.unet.layers[9].trainable = False
    rim.unet.layers[15].trainable = False  # bottleneck GRU
    # output layer
    # rim.unet.layers[-2].trainable = False
    # input layer
    # rim.unet.layers[-1].trainable = False
    # decoding layers
    rim.unet.layers[10].trainable = False # L5
    # rim.unet.layers[11].trainable = False
    # rim.unet.layers[12].trainable = False
    # rim.unet.layers[13].trainable = False
    # rim.unet.layers[14].trainable = False # L1

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.experiment_name + "_" + args.model + "_" + args.dataset + f"_{THIS_WORKER:03d}.h5"), 'w') as hf:
        hf.create_dataset(name="observation", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        hf.create_dataset(name="psf",  shape=[data_len, 20, 20, 1], dtype=np.float32)
        hf.create_dataset(name="psf_fwhm", shape=[data_len], dtype=np.float32)
        hf.create_dataset(name="noise_rms", shape=[data_len], dtype=np.float32)
        hf.create_dataset(name="source", shape=[data_len, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="kappa", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="observation_pred", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        hf.create_dataset(name="observation_pred_reoptimized", shape=[data_len, phys.pixels, phys.pixels, 1], dtype=np.float32)
        hf.create_dataset(name="source_pred", shape=[data_len, rim.steps, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="source_pred_reoptimized", shape=[data_len, phys.src_pixels, phys.src_pixels, 1])
        hf.create_dataset(name="kappa_pred", shape=[data_len, rim.steps, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="kappa_pred_reoptimized", shape=[data_len, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="chi_squared", shape=[data_len, rim.steps], dtype=np.float32)
        hf.create_dataset(name="chi_squared_reoptimized", shape=[data_len], dtype=np.float32)
        hf.create_dataset(name="chi_squared_reoptimized_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
        hf.create_dataset(name="source_optim_mse", shape=[data_len], dtype=np.float32)
        hf.create_dataset(name="source_optim_mse_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
        hf.create_dataset(name="kappa_optim_mse", shape=[data_len], dtype=np.float32)
        hf.create_dataset(name="kappa_optim_mse_series", shape=[data_len, args.re_optimize_steps], dtype=np.float32)
        hf.create_dataset(name="observation_coherence_spectrum", shape=[data_len, args.observation_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="source_coherence_spectrum",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="observation_coherence_spectrum2", shape=[data_len, args.observation_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="observation_coherence_spectrum_reoptimized", shape=[data_len, args.observation_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="source_coherence_spectrum2",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="source_coherence_spectrum_reoptimized",  shape=[data_len, args.source_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="kappa_coherence_spectrum", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="kappa_coherence_spectrum_reoptimized", shape=[data_len, args.kappa_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="observation_frequencies", shape=[args.observation_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="source_frequencies", shape=[args.source_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="kappa_frequencies", shape=[args.kappa_coherence_bins], dtype=np.float32)
        hf.create_dataset(name="kappa_fov", shape=[1], dtype=np.float32)
        hf.create_dataset(name="source_fov", shape=[1], dtype=np.float32)
        hf.create_dataset(name="observation_fov", shape=[1], dtype=np.float32)
        for batch, j in enumerate(range((THIS_WORKER-1) * data_len, THIS_WORKER * data_len)):
            b = j // B
            k = j % B
            observation = dataset[b]["observation"][k][None, ...]
            source = dataset[b]["source"][k][None, ...]
            kappa = dataset[b]["kappa"][k][None, ...]
            noise_rms = np.array([dataset[b]["noise_rms"][k]])
            psf = dataset[b]["psf"][k][None, ...]
            fwhm = dataset[b]["psf_fwhm"][k]

            checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()  # reset model weights
            # Compute predictions for kappa and source
            source_pred, kappa_pred, chi_squared = rim.predict(observation, noise_rms, psf)
            observation_pred = phys.forward(source_pred[-1], kappa_pred[-1], psf)
            # reset the seed for reproducible sampling in the VAE for EWC
            tf.random.set_seed(args.seed)
            np.random.seed(args.seed)
            # Initialize regularization term
            ewc = EWC(
                observation=observation,
                noise_rms=noise_rms,
                psf=psf,
                phys=phys,
                rim=rim,
                source_vae=source_vae,
                kappa_vae=kappa_vae,
                n_samples=args.sample_size,
                sigma_source=args.source_vae_ball_size,
                sigma_kappa=args.kappa_vae_ball_size
            )
            # Re-optimize weights of the model
            STEPS = args.re_optimize_steps
            learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=args.learning_rate,
                decay_rate=args.decay_rate,
                decay_steps=args.decay_steps,
                staircase=args.staircase
            )
            optim = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedule)

            chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
            source_mse = tf.TensorArray(DTYPE, size=STEPS)
            kappa_mse = tf.TensorArray(DTYPE, size=STEPS)
            best = chi_squared[-1, 0]
            source_best = source_pred[-1]
            kappa_best = kappa_pred[-1]
            source_mse_best = tf.reduce_mean((source_best - rim.source_inverse_link(source)) ** 2)
            kappa_mse_best = tf.reduce_sum(wk(kappa) * (kappa_best - rim.kappa_inverse_link(kappa)) ** 2)

            for current_step in tqdm(range(STEPS)):
                with tf.GradientTape() as tape:
                    tape.watch(unet.trainable_variables)
                    s, k, chi_sq = rim.call(observation, noise_rms, psf, outer_tape=tape)
                    cost = tf.reduce_mean(chi_sq)  # mean over time steps
                    cost += tf.reduce_sum(rim.unet.losses)  # L2 regularisation
                    cost += args.lam_ewc * ewc.penalty(rim)  # Elastic Weights Consolidation

                log_likelihood = chi_sq[-1]
                chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood)
                source_o = s[-1]
                kappa_o = k[-1]
                source_mse = source_mse.write(index=current_step, value=tf.reduce_mean((source_o - rim.source_inverse_link(source)) ** 2))
                kappa_mse = kappa_mse.write(index=current_step, value=tf.reduce_sum(wk(kappa) * (kappa_o - rim.kappa_inverse_link(kappa)) ** 2))
                if 2 * chi_sq[-1, 0] < 1.0 and args.early_stopping:
                    source_best = rim.source_link(source_o)
                    kappa_best = rim.kappa_link(kappa_o)
                    best = chi_sq[-1, 0]
                    source_mse_best = tf.reduce_mean((source_o - rim.source_inverse_link(source)) ** 2)
                    kappa_mse_best = tf.reduce_sum(wk(kappa) * (kappa_o - rim.kappa_inverse_link(kappa)) ** 2)
                    break
                if chi_sq[-1, 0] < best:
                    source_best = rim.source_link(source_o)
                    kappa_best = rim.kappa_link(kappa_o)
                    best = chi_sq[-1, 0]
                    source_mse_best = tf.reduce_mean((source_o - rim.source_inverse_link(source)) ** 2)
                    kappa_mse_best = tf.reduce_sum(wk(kappa) * (kappa_o - rim.kappa_inverse_link(kappa)) ** 2)

                grads = tape.gradient(cost, unet.trainable_variables)
                optim.apply_gradients(zip(grads, unet.trainable_variables))

            source_o = source_best
            kappa_o = kappa_best
            y_pred = phys.forward(source_o, kappa_o, psf)
            chi_sq_series = tf.transpose(chi_squared_series.stack(), perm=[1, 0])
            source_mse = source_mse.stack()[None, ...]
            kappa_mse = kappa_mse.stack()[None, ...]

            # Compute Power spectrum of converged predictions
            _ps_observation = ps_observation.cross_correlation_coefficient(observation[..., 0], observation_pred[..., 0])
            _ps_observation2 = ps_observation.cross_correlation_coefficient(observation[..., 0], y_pred[..., 0])
            _ps_kappa = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_pred[-1])[..., 0])
            _ps_kappa2 = ps_kappa.cross_correlation_coefficient(log_10(kappa)[..., 0], log_10(kappa_o[..., 0]))
            _ps_source = ps_source.cross_correlation_coefficient(source[..., 0], source_pred[-1][..., 0])
            _ps_source2 = ps_source.cross_correlation_coefficient(source[..., 0], source_o[..., 0])

            # save results
            hf["observation"][batch] = observation.astype(np.float32)
            hf["psf"][batch] = psf.astype(np.float32)
            hf["psf_fwhm"][batch] = fwhm
            hf["noise_rms"][batch] = noise_rms.astype(np.float32)
            hf["source"][batch] = source.astype(np.float32)
            hf["kappa"][batch] = kappa.astype(np.float32)
            hf["observation_pred"][batch] = observation_pred.numpy().astype(np.float32)
            hf["observation_pred_reoptimized"][batch] = y_pred.numpy().astype(np.float32)
            hf["source_pred"][batch] = tf.transpose(source_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
            hf["source_pred_reoptimized"][batch] = source_o.numpy().astype(np.float32)
            hf["kappa_pred"][batch] = tf.transpose(kappa_pred, perm=(1, 0, 2, 3, 4)).numpy().astype(np.float32)
            hf["kappa_pred_reoptimized"][batch] = kappa_o.numpy().astype(np.float32)
            hf["chi_squared"][batch] = 2*tf.transpose(chi_squared).numpy().astype(np.float32)
            hf["chi_squared_reoptimized"][batch] = 2*best.numpy().astype(np.float32)
            hf["chi_squared_reoptimized_series"][batch] = 2*chi_sq_series.numpy().astype(np.float32)
            hf["source_optim_mse"][batch] = source_mse_best.numpy().astype(np.float32)
            hf["source_optim_mse_series"][batch] = source_mse.numpy().astype(np.float32)
            hf["kappa_optim_mse"][batch] = kappa_mse_best.numpy().astype(np.float32)
            hf["kappa_optim_mse_series"][batch] = kappa_mse.numpy().astype(np.float32)
            hf["observation_coherence_spectrum"][batch] = _ps_observation
            hf["observation_coherence_spectrum_reoptimized"][batch] = _ps_observation2
            hf["source_coherence_spectrum"][batch] = _ps_source
            hf["source_coherence_spectrum_reoptimized"][batch] = _ps_source2
            hf["kappa_coherence_spectrum"][batch] = _ps_kappa
            hf["kappa_coherence_spectrum_reoptimized"][batch] = _ps_kappa2

            if batch == 0:
                _, f = np.histogram(np.fft.fftfreq(phys.pixels)[:phys.pixels//2], bins=ps_observation.bins)
                f = (f[:-1] + f[1:]) / 2
                hf["observation_frequencies"][:] = f
                _, f = np.histogram(np.fft.fftfreq(phys.src_pixels)[:phys.src_pixels//2], bins=ps_source.bins)
                f = (f[:-1] + f[1:]) / 2
                hf["source_frequencies"][:] = f
                _, f = np.histogram(np.fft.fftfreq(phys.kappa_pixels)[:phys.kappa_pixels//2], bins=ps_kappa.bins)
                f = (f[:-1] + f[1:]) / 2
                hf["kappa_frequencies"][:] = f
                hf["kappa_fov"][0] = phys.kappa_fov
                hf["source_fov"][0] = phys.src_fov


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="finetune_likelihood_with_ewc")
    parser.add_argument("--model",              required=True,      help="Model to get predictions from")
    parser.add_argument("--source_vae",         required=True)
    parser.add_argument("--kappa_vae",          required=True)
    parser.add_argument("--h5_pattern",         required=True)
    parser.add_argument("--dataset",            required=True)
    parser.add_argument("--sample_size",        default=200,        type=int, help="Number of VAE sampled required to compute the Fisher diagonal")
    parser.add_argument("--buffer_size",        default=10000,      type=int)
    parser.add_argument("--observation_coherence_bins",    default=40,     type=int)
    parser.add_argument("--source_coherence_bins",  default=40,     type=int)
    parser.add_argument("--kappa_coherence_bins",   default=40,     type=int)
    parser.add_argument("--re_optimize_steps",  default=2000,       type=int)
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=1,          type=float)
    parser.add_argument("--decay_steps",        default=1000,         type=float)
    parser.add_argument("--staircase",          action="store_true")
    parser.add_argument("--early_stopping",     action="store_true")
    parser.add_argument("--seed",               default=42,         type=int)
    parser.add_argument("--l2_amp",             default=0,          type=float)
    parser.add_argument("--lam_ewc",            default=1,       type=float)
    parser.add_argument("--source_vae_ball_size",   default=0.5,    type=float, help="Standard deviation of the source VAE latent space sampling around RIM prediction")
    parser.add_argument("--kappa_vae_ball_size",    default=0.5,    type=float, help="Standard deviation of the kappa VAE latent space sampling around RIM prediction")

    args = parser.parse_args()
    distributed_strategy(args)
