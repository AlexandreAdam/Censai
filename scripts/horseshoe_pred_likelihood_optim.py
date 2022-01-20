from censai import RIMSharedUnetv3, PhysicalModelv2
from censai.models import SharedUnetModelv4
import tensorflow as tf
import json, os
from astropy.io import fits
from tqdm import tqdm
from censai.definitions import DTYPE
import h5py
import numpy as np


def predict(self, observation, noise_rms, psf, rim_input, mask):#, lens_light):
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
    for current_step in tqdm(range(self.steps-1)):
        with tf.GradientTape() as g:
            g.watch(source)
            g.watch(kappa)
            y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
            log_likelihood = 0.5 * tf.reduce_sum(mask * tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = log_likelihood
        source_grad, kappa_grad = g.gradient(cost, [source, kappa])
        source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
        source, kappa, states = self.time_step(rim_input, source, kappa, source_grad, kappa_grad, states, training=False)
        source_series = source_series.write(index=current_step+1, value=self.source_link(source))
        kappa_series = kappa_series.write(index=current_step+1, value=self.kappa_link(kappa))
        obs_series = obs_series.write(index=current_step, value=y_pred)
        chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood/self.pixels**2)
    # last step score
    y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
    obs_series = obs_series.write(index=self.steps-1, value=y_pred)
    log_likelihood = 0.5 * tf.reduce_sum(mask * tf.square(y_pred - observation) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
    chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood/self.pixels**2)
    return source_series.stack(), kappa_series.stack(), chi_squared_series.stack(), obs_series.stack()


def main(args):
    data = fits.open(args.rim_data)
    observation = tf.constant(data["preprocessed_observation"], dtype=tf.float32)[None, ..., None]  # galfitted, noise padded, normalized
    # lens_light = tf.constant(data["lens_light"], dtype=tf.float32)[None, ..., None]
    rim_input = tf.constant(data["rim_input"].data, dtype=tf.float32)[None, ..., None]
    psf = tf.constant(data["psf"], dtype=tf.float32)[None, ..., None]
    noise_rms = tf.constant(data["PRIMARY"].header["noiserms"])[None]
    image_fov = data["PRIMARY"].header["fov"]
    src_fov = data["PRIMARY"].header["srcfov"]
    mask = tf.constant(data["mask"], dtype=tf.float32)[None, ..., None]
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

    for current_step in tqdm(range(STEPS)):
        with tf.GradientTape() as tape:
            tape.watch(unet.trainable_variables)
            source_o, kappa_o, chi_sq, y_pred_o = predict(rim, observation, noise_rms, psf, rim_input, mask)
            cost = tf.reduce_mean(chi_sq)  # mean over time steps
            cost += tf.reduce_sum(rim.unet.losses)

        chi_squared_series = chi_squared_series.write(index=current_step, value=2*chi_sq[-1, 0])

        grads = tape.gradient(cost, unet.trainable_variables)
        optim.apply_gradients(zip(grads, unet.trainable_variables))

        if 2 * chi_sq[-1, 0] < 1.:
            source_best = source_o
            kappa_best = kappa_o
            best = chi_sq[:, 0]
            obs_best = y_pred_o
            break
        if chi_sq[-1, 0] < best[-1]:
            source_best = source_o
            kappa_best = kappa_o
            obs_best = y_pred_o
            best = chi_sq[:, 0]

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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--experiment_name",    default="horseshoe")
    parser.add_argument("--rim_data",           required=True,      help="See horseshoe jupyter")
    parser.add_argument("--model",              required=True,      help="RIM")
    parser.add_argument("--reoptimize_steps",   default=3000,       type=int)
    parser.add_argument("--learning_rate",      default=1e-6,       type=float)
    parser.add_argument("--l2_amp",             default=1e-6,       type=float)
    parser.add_argument("--decay_rate",         default=1,          type=float)
    parser.add_argument("--decay_steps",        default=50,         type=float)
    parser.add_argument("--staircase",          action="store_true")

    args = parser.parse_args()
    main(args)
