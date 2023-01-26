from censai import RIM, PhysicalModel, EWC
from censai.models import Model, VAE
import tensorflow as tf
import json, os
from astropy.io import fits
from censai.definitions import DTYPE
import h5py
import numpy as np
from tqdm import tqdm
from censai.utils import nulltape


def call(self, observation, noise_rms, psf, rim_input, outer_tape=nulltape):
    """
    Used for training. Return kappa and source maps in model space.
    """
    batch_size = observation.shape[0]
    source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
    source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states)  # Use rim_input to make an initial guess at the solution
    source_series = tf.TensorArray(DTYPE, size=self.steps)
    kappa_series = tf.TensorArray(DTYPE, size=self.steps)
    chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
    #  Record initial guess
    source_series = source_series.write(index=0, value=source)
    kappa_series = kappa_series.write(index=0, value=kappa)
    #  Main optimization loop
    for current_step in range(self.steps - 1):
        with outer_tape.stop_recording():
            with tf.GradientTape() as g:
                g.watch(source)
                g.watch(kappa)
                y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
                log_likelihood = 0.5 * tf.reduce_sum(
                    tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
                cost = log_likelihood
        source_grad, kappa_grad = g.gradient(cost, [source, kappa])
        source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
        source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states, training=False)
        source_series = source_series.write(index=current_step + 1, value=source)
        kappa_series = kappa_series.write(index=current_step + 1, value=kappa)
        chi_squared_series = chi_squared_series.write(index=current_step,
                                                      value=2 * log_likelihood / self.physical_model.pixels ** 2)
    #  Last step score
    y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
    chi_squared_series = chi_squared_series.write(index=self.steps - 1,
                                                  value=2 * log_likelihood / self.physical_model.pixels ** 2)
    return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()  # stack along 0-th dimension


def predict(self: RIM, observation, noise_rms, psf, rim_input):
    """
    Used in inference. Return physical kappa and source maps.
    """
    batch_size = observation.shape[0]
    source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
    source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad,
                                           states)  # Use lens to make an initial guess with Unet
    source_series = tf.TensorArray(DTYPE, size=self.steps)
    kappa_series = tf.TensorArray(DTYPE, size=self.steps)
    chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
    obs_series = tf.TensorArray(DTYPE, size=self.steps)

    # record initial guess
    source_series = source_series.write(index=0, value=self.source_link(source))
    kappa_series = kappa_series.write(index=0, value=self.kappa_link(kappa))
    # Main optimization loop
    for current_step in range(self.steps - 1):
        with tf.GradientTape() as g:
            g.watch(source)
            g.watch(kappa)
            y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
            log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = log_likelihood
        source_grad, kappa_grad = g.gradient(cost, [source, kappa])
        source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
        source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states,
                                               training=False)
        source_series = source_series.write(index=current_step + 1, value=self.source_link(source))
        kappa_series = kappa_series.write(index=current_step + 1, value=self.kappa_link(kappa))
        obs_series = obs_series.write(index=current_step, value=y_pred)
        chi_squared_series = chi_squared_series.write(index=current_step,
                                                      value=2 * log_likelihood / self.physical_model.pixels ** 2)
    # last step score
    y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
    obs_series = obs_series.write(index=self.steps - 1, value=y_pred)
    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
    chi_squared_series = chi_squared_series.write(index=self.steps - 1,
                                                  value=2 * log_likelihood / self.physical_model.pixels ** 2)
    return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()


def main(args):
    data = fits.open(args.rim_data)
    observation = tf.constant(data["preprocessed_observation"].data, dtype=tf.float32)[None, ..., None]  # galfitted, noise padded, normalized
    rim_input = tf.constant(data["rim_input"].data, dtype=tf.float32)[None, ..., None]
    psf = tf.constant(data["psf"].data, dtype=tf.float32)[None, ..., None]
    noise_rms = tf.constant(data["PRIMARY"].header["noiserms"])[None]
    image_fov = data["PRIMARY"].header["CDELT1"] * 3600 * observation.shape[1] # convert pixel size to arcsec

    phys = PhysicalModel(
        pixels=observation.shape[1],
        kappa_pixels=128,
        src_pixels=128,
        image_fov=image_fov,
        kappa_fov=image_fov,
        src_fov=args.src_fov,
        method="fft",
    )

    phys_sample = PhysicalModel(  # used for EWC
        pixels=128,
        kappa_pixels=128,
        src_pixels=128,
        image_fov=7.69,
        kappa_fov=7.69,
        src_fov=3,
        method="fft",
    )

    path = os.path.join(os.getenv("CENSAI_PATH"), "models", args.model)
    with open(os.path.join(path, "unet_hparams.json")) as f:
        unet_params = json.load(f)
    unet_params["kernel_l2_amp"] = args.l2_amp
    unet = Model(**unet_params)
    ckpt = tf.train.Checkpoint(net=unet)
    checkpoint_manager = tf.train.CheckpointManager(ckpt, path, 1)
    checkpoint_manager.checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
    with open(os.path.join(path, "rim_hparams.json")) as f:
        rim_params = json.load(f)
    rim_params["source_link"] = "relu"
    rim = RIM(phys, unet, **rim_params)
    rim_sample = RIM(phys_sample, unet, **rim_params)

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

    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        decay_steps=args.decay_steps,
        staircase=args.staircase
    )
    optim = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_schedule)

    source, kappa, chisq_baseline = predict(rim, observation, noise_rms, psf, rim_input)

    STEPS = args.reoptimize_steps
    chi_squared_series = []
    best = chisq_baseline[-1, 0]
    source_best = source[-1]
    kappa_best = kappa[-1]

    # ===================== Compute the Fisher ==============================
    # reset the seed for reproducible sampling in the VAE for EWC
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    # Initialize regularization term
    ewc = EWC(
        observation=rim_input,
        noise_rms=noise_rms,
        psf=psf,
        phys=phys_sample,
        rim=rim_sample,  # used to compute the fisher matrix
        source_vae=source_vae,
        kappa_vae=kappa_vae,
        n_samples=args.n_samples,
        sigma_source=args.source_ball_size,
        sigma_kappa=args.kappa_ball_size
    )

    with h5py.File(os.path.join(os.getenv("CENSAI_PATH"), "results", args.experiment_name + "_" + args.model + ".h5"), 'w') as hf:
        hf.create_dataset(name="source_pred", shape=[STEPS // args.save_every, phys.src_pixels, phys.src_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="kappa_pred", shape=[STEPS // args.save_every, phys.kappa_pixels, phys.kappa_pixels, 1], dtype=np.float32)
        hf.create_dataset(name="observation_pred", shape=[STEPS // args.save_every, phys.pixels, phys.pixels, 1], dtype=np.float32)
        hf.create_dataset(name="chi_squared_pred", shape=[STEPS // args.save_every], dtype=np.float32)

        pbar = tqdm(range(STEPS))
        for current_step in pbar:
            with tf.GradientTape() as tape:
                tape.watch(unet.trainable_variables)
                s, k, chi_sq = call(rim, observation, noise_rms, psf, rim_input, outer_tape=tape)
                cost = tf.reduce_mean(chi_sq)  # mean over time steps
                cost += tf.reduce_sum(rim.unet.losses)  # L2 regularisation
                cost += args.lam_ewc * ewc.penalty(rim)

            pbar.set_description(f"chisq = {chi_sq[-1, 0].numpy():.1f}")
            chi_squared_series.append(chi_sq[-1, 0])
            source_o = rim.source_link(s[-1])
            kappa_o = rim.kappa_link(k[-1])

            if current_step % args.save_every == 0:
                hf["source_pred"][current_step // args.save_every] = source_o
                hf["kappa_pred"][current_step // args.save_every] = kappa_o
                y_pred = phys.forward(source_o, kappa_o, psf)
                hf["observation_pred"][current_step // args.save_every] = y_pred

            # early stopping
            # if chi_sq[-1, 0] < 1.0:
            #     source_best = rim.source_link(source_o)
            #     kappa_best = rim.kappa_link(kappa_o)
            #     best = chi_sq[-1, 0]
            #     break
            if chi_sq[-1, 0] < best:
                source_best = source_o
                kappa_best = kappa_o
                best = chi_sq[-1, 0]

            grads = tape.gradient(cost, unet.trainable_variables)
            optim.apply_gradients(zip(grads, unet.trainable_variables))

        # Record best prediction
        y_pred = phys.forward(source_best, kappa_best, psf)
        hf["source_baseline"] = source.numpy().squeeze().astype(np.float32)
        hf["kappa_baseline"] = kappa.numpy().squeeze().astype(np.float32)
        hf["source_best"] = source_best.numpy().squeeze().astype(np.float32)
        hf["kappa_best"] = kappa_best.numpy().squeeze().astype(np.float32)
        hf["observation_pred_best"] = y_pred.numpy().squeeze().astype(np.float32)
        hf["chi_squared_best"] = np.array([best]).astype(np.float32)
        hf["chi_squared_series"] = np.array(chi_squared_series).astype(np.float32)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--rim_data", required=True)
    parser.add_argument("--model", required=True, help="RIM")
    parser.add_argument("--source_vae", required=True)
    parser.add_argument("--kappa_vae", required=True)
    parser.add_argument("--n_samples", default=200, type=int)
    parser.add_argument("--source_ball_size", default=0.5, type=float)
    parser.add_argument("--kappa_ball_size", default=0.5, type=float)
    parser.add_argument("--reoptimize_steps", default=10000, type=int)
    parser.add_argument("--learning_rate", default=1e-6, type=float)
    parser.add_argument("--lam_ewc", default=2e5, type=float)
    parser.add_argument("--l2_amp", default=0, type=float)
    parser.add_argument("--decay_rate", default=1, type=float)
    parser.add_argument("--decay_steps", default=50, type=float)
    parser.add_argument("--staircase", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_every", default=1000, type=int, help="Save every x steps.")
    parser.add_argument("--src_fov", default=3, type=float)

    args = parser.parse_args()
    main(args)
