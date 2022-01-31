from censai import RIMSharedUnetv3, PhysicalModelv2
from censai.models import SharedUnetModelv4, VAE
import tensorflow as tf
import json, os
from astropy.io import fits
from tqdm import tqdm
from censai.definitions import DTYPE, log_10
import h5py
import numpy as np


def predict(self:RIMSharedUnetv3, observation, noise_rms, psf, rim_input):#, mask):#, lens_light):
    """
    Used in inference. Return physical kappa and source maps.
    """
    batch_size = observation.shape[0]
    source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
    source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states)  # Use lens to make an initial guess with Unet
    source_series = tf.TensorArray(DTYPE, size=self.steps)
    kappa_series = tf.TensorArray(DTYPE, size=self.steps)
    chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
    obs_series = tf.TensorArray(DTYPE, size=self.steps)

    # record initial guess
    source_series = source_series.write(index=0, value=self.source_link(source))
    kappa_series = kappa_series.write(index=0, value=self.kappa_link(kappa))
    # Main optimization loop
    for current_step in range(self.steps-1):
        with tf.GradientTape() as g:
            g.watch(source)
            g.watch(kappa)
            y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
            log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = log_likelihood
        source_grad, kappa_grad = g.gradient(cost, [source, kappa])
        source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
        source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states, training=False)
        source_series = source_series.write(index=current_step+1, value=self.source_link(source))
        kappa_series = kappa_series.write(index=current_step+1, value=self.kappa_link(kappa))
        obs_series = obs_series.write(index=current_step, value=y_pred)
        chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood/self.physical_model.pixels**2)
    # last step score
    y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
    obs_series = obs_series.write(index=self.steps-1, value=y_pred)
    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
    chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood/self.physical_model.pixels**2)
    return source_series.stack(), kappa_series.stack(), chi_squared_series.stack(), obs_series.stack()


def main(args):
    data = fits.open(args.rim_data)
    observation = tf.constant(data["preprocessed_observation"].data, dtype=tf.float32)[None, ..., None]  # galfitted, noise padded, normalized
    # lens_light = tf.constant(data["lens_light"], dtype=tf.float32)[None, ..., None]
    rim_input = tf.constant(data["rim_input"].data, dtype=tf.float32)[None, ..., None]
    psf = tf.constant(data["psf"].data, dtype=tf.float32)[None, ..., None]
    noise_rms = tf.constant(data["PRIMARY"].header["noiserms"])[None]
    image_fov = data["PRIMARY"].header["fov"]
    src_fov = data["PRIMARY"].header["srcfov"]
    mask = tf.constant(data["mask"].data, dtype=tf.float32)[None, ..., None]
    # sigma_image = tf.constant(data["sigma_image"], dtype=tf.float32)[None, ..., None]

    phys = PhysicalModelv2(  # use same parameter as was used during training, rescale kappa later
        pixels=observation.shape[1],
        kappa_pixels=128,
        src_pixels=128,
        image_fov=image_fov,
        kappa_fov=image_fov,
        src_fov=src_fov,
        method="fft",
    )

    phys_sample = PhysicalModelv2(  # use same parameter as was used during training, rescale kappa later
        pixels=128,
        kappa_pixels=128,
        src_pixels=128,
        image_fov=image_fov,
        kappa_fov=image_fov,
        src_fov=src_fov,
        method="fft",
    )

    path = os.path.join(os.getenv("CENSAI_PATH"), "models", args.model)
    with open(os.path.join(path, "unet_hparams.json")) as f:
        unet_params = json.load(f)
    unet_params["kernel_l2_amp"] = args.l2_amp
    unet = SharedUnetModelv4(**unet_params)
    ckpt = tf.train.Checkpoint(net=unet)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, path, 1)
    checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    with open(os.path.join(path, "rim_hparams.json")) as f:
        rim_params = json.load(f)
    rim_params["source_link"] = "relu"
    rim = RIMSharedUnetv3(phys, unet, **rim_params)
    rim_sample = RIMSharedUnetv3(phys_sample, unet, **rim_params)

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

    source, kappa, chisq, y_pred = predict(rim, observation, noise_rms, psf, rim_input)

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)

    STEPS = args.reoptimize_steps
    chi_squared_series = tf.TensorArray(DTYPE, size=STEPS)
    best = chisq[:, 0]
    source_best = source
    kappa_best = kappa
    obs_best = y_pred

    y_mean = tf.zeros_like(y_pred[-1])
    y_var = tf.zeros_like(y_mean)
    # prediction and uncertainty collected in model space
    source_mean = tf.zeros_like(source[-1])
    kappa_mean = tf.zeros_like(kappa[-1])
    source_var = tf.zeros_like(source_mean)
    kappa_var = tf.zeros_like(kappa_mean)
    chi_squared_series_mean_pred = tf.TensorArray(DTYPE, size=STEPS - args.burn_in)

    # ===================== VAE SAMPLING from PRIOR ==============================
    # Latent code of model predictions
    z_source, _ = source_vae.encoder(source)
    z_kappa, _ = kappa_vae.encoder(log_10(kappa))

    z_source_std = args.source_vae_ball_size
    z_kappa_std = args.kappa_vae_ball_size

    # Sample latent code, then decode and forward
    z_s = tf.random.normal(shape=[args.sample_size, source_vae.latent_size], mean=z_source, stddev=z_source_std)
    z_k = tf.random.normal(shape=[args.sample_size, kappa_vae.latent_size], mean=z_kappa, stddev=z_kappa_std)
    sampled_source = tf.nn.relu(source_vae.decode(z_s))
    sampled_source /= tf.reduce_max(sampled_source, axis=(1, 2, 3), keepdims=True)
    sampled_kappa = kappa_vae.decode(z_k)  # output in log_10 space
    sampled_lens = phys_sample.noisy_forward(sampled_source, 10 ** sampled_kappa, noise_rms, tf.tile(psf, [args.sample_size, 1, 1, 1]))
    wk = tf.keras.layers.Lambda(lambda k: tf.sqrt(k) / tf.reduce_sum(tf.sqrt(k), axis=(1, 2, 3), keepdims=True))

    for current_step in tqdm(range(STEPS)):
        with tf.GradientTape() as tape:
            tape.watch(unet.trainable_variables)
            s, k, chi_sq = rim_sample.predict(sampled_lens, noise_rms, tf.tile(psf, [args.sample_size, 1, 1, 1]))
            _kappa_mse = tf.reduce_sum(wk(10 ** sampled_kappa) * (k - sampled_kappa) ** 2, axis=(2, 3, 4))
            cost = tf.reduce_mean(_kappa_mse)
            cost += tf.reduce_mean((s - sampled_source) ** 2)
            cost += tf.reduce_sum(rim.unet.losses)  # weight decay

        grads = tape.gradient(cost, unet.trainable_variables)
        optim.apply_gradients(zip(grads, unet.trainable_variables))

        # Check prediction on horseshoe
        source_o, kappa_o, chi_sq, y_pred_o = predict(rim, observation, noise_rms, psf, rim_input)
        chi_squared_series = chi_squared_series.write(index=current_step, value=2*chi_sq[-1, 0])

        if chi_sq[-1, 0] < best[-1]:
            source_best = source_o
            kappa_best = kappa_o
            obs_best = y_pred_o
            best = chi_sq[:, 0]

        # Welford's online algorithm for moving variance
        if current_step >= args.burn_in:
            step = current_step - args.burn_in
            # source
            delta = source_o[-1] - source_mean
            source_mean = (step * source_mean + source_o[-1]) / (step + 1)
            delta2 = source_o[-1] - source_mean
            source_var += delta * delta2
            # kappa
            delta = kappa_o[-1] - kappa_mean
            kappa_mean = (step * kappa_mean + kappa_o) / (step + 1)
            delta2 = kappa_o[-1] - kappa_mean
            kappa_var += delta * delta2
            # observation
            y_o = phys.forward(source_o[-1], kappa_o[-1], psf)
            delta = y_o - y_mean
            y_mean = (step * y_mean + y_o) / (step + 1)
            delta2 = y_o - y_mean
            y_var += delta * delta2
            chisq_mean = tf.reduce_mean((y_mean - observation) ** 2 / noise_rms[:, None, None, None] ** 2)
            chi_squared_series_mean_pred = chi_squared_series_mean_pred.write(index=current_step - args.burn_in, value=chisq_mean)

    source_var /= STEPS - args.burn_in
    kappa_var /= STEPS - args.burn_in
    y_var /= STEPS - args.burn_in
    y_mean_pred = phys.forward(source_mean, kappa_mean, psf)
    chisq_mean_pred = tf.reduce_mean((y_mean_pred - observation) ** 2 / noise_rms[:, None, None, None] ** 2)
    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.experiment_name + "_" + args.model + ".h5"), 'w') as hf:
        hf["source_rim"] = source.numpy().squeeze().astype(np.float32)
        hf["kappa_rim"] = kappa.numpy().squeeze().astype(np.float32)
        hf["reconstruction_rim"] = y_pred.numpy().squeeze().astype(np.float32)
        hf["chisq_rim"] = chisq
        hf["source_rim_sgd"] = source_best.numpy().squeeze().astype(np.float32)
        hf["kappa_rim_sgd"] = kappa_best.numpy().squeeze().astype(np.float32)
        hf["reconstruction_rim_sgd"] = obs_best.numpy().squeeze().astype(np.float32)
        hf["chisq_rim_sgd"] = best
        hf["chi_squared_series"] = chi_squared_series.stack().numpy().squeeze().astype(np.float32)
        hf["source_mean"] = source_mean.numpy().squeeze().astype(np.float32)
        hf["source_var"] = source_var.numpy().squeeze().astype(np.float32)
        hf["kappa_mean"] = kappa_mean.numpy().squeeze().astype(np.float32)
        hf["kappa_vae"] = kappa_var.numpy().squeeze().astype(np.float32)
        hf["y_mean"] = y_mean.numpy().squeeze().astype(np.float32)
        hf["y_mean_pred"] = y_mean_pred.numpy().squeeze().astype(np.float32)
        hf["chi_squared_y_mean_pred"] = chisq_mean_pred
        hf["y_var"] = y_var.numpy().squeeze().astype(np.float32)
        hf["chi_squared_mean_series"] = chi_squared_series_mean_pred.stack().numpy().squeeze().astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="horseshoe")
    parser.add_argument("--rim_data",           required=True,      help="See horseshoe jupyter")
    parser.add_argument("--model",              required=True,      help="RIM")
    parser.add_argument("--source_vae",         required=True)
    parser.add_argument("--kappa_vae",          required=True)
    parser.add_argument("--sample_size",        default=10,         type=int)
    parser.add_argument("--source_vae_ball_size",   default=0.5,    type=float, help="Standard deviation of the source VAE latent space sampling around RIM prediction")
    parser.add_argument("--kappa_vae_ball_size",    default=0.5,    type=float, help="Standard deviation of the kappa VAE latent space sampling around RIM prediction")
    parser.add_argument("--reoptimize_steps",   default=3000,       type=int)
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--l2_amp",             default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=1,          type=float)
    parser.add_argument("--decay_steps",        default=50,         type=float)
    parser.add_argument("--staircase",          action="store_true")
    parser.add_argument("--burn_in",            default=2500,       type=int)

    args = parser.parse_args()
    main(args)
