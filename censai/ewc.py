"""
Elastic Weight Compression Regularisation (Kirkpatric 2016)
"""
from censai import PhysicalModelv2 as PhysicalModel, RIMSharedUnetv3 as RIM
from censai.models import VAE
from censai.definitions import log_10
import tensorflow as tf
from copy import deepcopy


class EWC:
    def __init__(
            self,
            observation,
            noise_rms,
            psf,
            phys:PhysicalModel,
            rim: RIM,
            source_vae: VAE,
            kappa_vae: VAE,
            n_samples=100,
            sigma_source=0.5,
            sigma_kappa=0.5
    ):
        """
        Make a copy of initial parameters \varphi^{(0)} and compute the Fisher diagonal F_{ii}
        """
        wk = tf.keras.layers.Lambda(lambda k: tf.sqrt(k) / tf.reduce_sum(tf.sqrt(k), axis=(1, 2, 3), keepdims=True))
        # Baseline prediction from observation
        source_pred, kappa_pred, chi_squared = rim.predict(observation, noise_rms, psf)
        # Latent code of model predictions
        z_source, _ = source_vae.encoder(source_pred[-1])
        z_kappa, _ = kappa_vae.encoder(log_10(kappa_pred[-1]))
        # Deepcopy of the initial parameters
        self.initial_params = [deepcopy(w) for w in rim.unet.trainable_variables]
        self.fisher_diagonal = [tf.zeros_like(w) for w in self.initial_params]
        for n in range(n_samples):
            # Sample latent code around the prediction mean
            z_s = tf.random.normal(shape=[1, source_vae.latent_size], mean=z_source, stddev=sigma_source)
            z_k = tf.random.normal(shape=[1, kappa_vae.latent_size], mean=z_kappa, stddev=sigma_kappa)
            # Decode
            sampled_source = tf.nn.relu(source_vae.decode(z_s))
            sampled_source /= tf.reduce_max(sampled_source, axis=(1, 2, 3), keepdims=True)
            sampled_kappa = kappa_vae.decode(z_k)  # output in log_10 space
            # Simulate observation
            sampled_observation = phys.noisy_forward(sampled_source, 10 ** sampled_kappa, noise_rms, psf)
            # Compute the gradient of the MSE
            with tf.GradientTape() as tape:
                tape.watch(rim.unet.trainable_variables)
                s, k, chi_squared = rim.call(sampled_observation, noise_rms, psf)
                # Remove the temperature from the loss when computing the Fisher: sum instead of mean, and weighted sum is renormalized by number of pixels
                _kappa_mse = phys.kappa_pixels**2*tf.reduce_sum(wk(10 ** sampled_kappa) * (k - sampled_kappa) ** 2, axis=(2, 3, 4))
                cost = tf.reduce_sum(_kappa_mse)
                cost += tf.reduce_sum((s - sampled_source) ** 2)
                # Fisher has tremendous sensibility to likelihood + related to our likelihood fine tuning, but it drowns information from the prior
                # cost += tf.reduce_mean(chi_squared)
            grad = tape.gradient(cost, rim.unet.trainable_variables)
            # Square the derivative relative to initial parameters and add to total
            self.fisher_diagonal = [F + g**2/n_samples for F, g in zip(self.fisher_diagonal, grad)]

    def penalty(self, rim):
        return 0.5*tf.reduce_sum([tf.reduce_sum(F * (varphi - varphi_0)**2) for (F, varphi, varphi_0)
                               in zip(self.fisher_diagonal, rim.unet.trainable_variables, self.initial_params)])
