import tensorflow as tf
from .decoder import Decoder
from .resnet_encoder import ResnetEncoder
from numpy import pi
from censai.galflow import convolve
from scipy.signal.windows import tukey
import numpy as np


class CosmosAutoencoder(tf.keras.Model):
    def __init__(
            self,
            pixels=128,  # side length of the input image, used to compute shape of bottleneck mainly
            layers=7,
            res_blocks_in_layer=2,
            conv_layers_per_block=2,
            filter_scaling=2,
            filter_init=8,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            activation="bipolar_relu",
            dropout_rate=None,
            batch_norm=False,
            latent_size=16
    ):
        super(CosmosAutoencoder, self).__init__()
        self.latent_size = latent_size
        self.encoder = ResnetEncoder(
            layers=layers,
            res_blocks_in_layer=res_blocks_in_layer,
            conv_layers_in_resblock=conv_layers_per_block,
            filter_scaling=filter_scaling,
            filters=filter_init,
            kernel_size=kernel_size,
            res_architecture=res_architecture,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            latent_size=latent_size,
            activation=activation
        )
        # compute size of mlp bottleneck from size of image and # of filters in the last encoding layer
        filters = filter_init*(int(filter_scaling**(layers)))
        pix = pixels//2**(layers)
        mlp_bottleneck = filters * pix**2
        self.decoder = Decoder(
            mlp_bottleneck=mlp_bottleneck,
            z_reshape_pix=pix,
            layers=layers,
            conv_layers=conv_layers_per_block,
            filter_scaling=filter_scaling,
            filters=filter_init,
            kernel_size=kernel_size,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation
        )

    def encode(self, x, psf):
        """

        Args:
            x: Batch of image of shape [None, pixels, pixels, 1]
            psf: Batch of the real FT of PSF associated with images, with noise padding, shape [None, 2*pixels, pixels+1, 1]

        Returns: Encoder latent vector z (shape [None, num_latent_variables])

        """
        input_shape = x.shape
        psf_image = tf.signal.irfft2d(tf.cast(psf[..., 0], tf.complex64))[..., tf.newaxis]
        # Roll the image to undo the fftshift, assuming x1 zero padding and x2 subsampling
        psf_image = tf.roll(psf_image, shift=[input_shape[1], input_shape[2]], axis=[1, 2])
        psf_image = tf.image.resize_with_crop_or_pad(psf_image, input_shape[1], input_shape[2])
        x = tf.concat([x, psf_image], axis=-1)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x, psf):
        return self.call(x, psf)

    def call(self, x, psf_image):
        x = tf.concat([x, psf_image], axis=-1)
        return self.decoder(self.encoder(x))

    def cost_function(self, x, psf, ps):
        """

        Args:
            x: Batch of image of shape [None, pixels, pixels, 1]
            psf: Batch of the real FT of PSF associated with images, with noise padding, shape [None, 2*pixels, pixels+1, 1]
            ps: Power spectrum of the noise, shape [None, pixels, pixels//2+1, 1], encoded as log(ps**2)

        Returns: Chi squared and added regularizer (except weight regularizers) -> scalar for each example

        """
        x_pred = self.decode(self.encode(x, psf))
        x_pred = convolve(x_pred, tf.cast(psf[..., 0], tf.complex64), zero_padding_factor=1)

        x = tf.signal.rfft2d(x[..., 0])
        x_pred = tf.signal.rfft2d(x_pred[..., 0])

        chi_squared = 0.5 * tf.reduce_mean(tf.abs((x - x_pred)**2 / tf.complex(tf.exp(ps) + 1e-8, 0.) / (2 * pi) ** 2), axis=[1, 2])
        return chi_squared

    def training_cost_function(self, image, psf, ps, skip_strength, l2_bottleneck, apodization_alpha, apodization_factor, tv_factor):
        """

        Args:
            x: Batch of image of shape [None, pixels, pixels, 1]
            psf: Batch of the real FT of PSF associated with images, with noise padding, shape [None, 2*pixels, pixels+1, 1]
            ps: Power spectrum of the noise, shape [None, pixels, pixels//2+1, 1], encoded as log(ps**2)
            skip_strength: Multiplicative factor applied to skip connections between encoder and decoder
            l2_bottleneck: l2 factor applied on identity loss in bottleneck
            apodization_alpha: Shape parameter of the Tukey window (Tapered cosine Window),
                    representing the fraction of the window inside the cosine tapered region.
                    If zero, the Tukey window is equivalent to a rectangular window.
                    If one, the Tukey window is equivalent to a Hann window.
            apodization_factor: Multiplicative factor applied on apodization loss
            tv_factor: Multiplicative factor applied on Total Variation (TV) loss

        Returns: Chi squared and added regularization loss

        """
        input_shape = image.shape
        psf_image = tf.signal.irfft2d(tf.cast(psf[..., 0], tf.complex64))[..., tf.newaxis]
        # Roll the image to undo the fftshift, assuming x1 zero padding and x2 subsampling
        psf_image = tf.roll(psf_image, shift=[input_shape[1], input_shape[2]], axis=[1, 2])
        psf_image = tf.image.resize_with_crop_or_pad(psf_image, input_shape[1], input_shape[2])
        x = tf.concat([image, psf_image], axis=-1)  # stack psf information in input

        z, skips = self.encoder.call_with_skip_connections(x)
        x_pred, bottleneck_l2_cost = self.decoder.call_with_skip_connections(z, skips, skip_strength, l2_bottleneck)
        x_pred = convolve(x_pred, tf.cast(psf[..., 0], tf.complex64), zero_padding_factor=1) # we already padded psf with noise in data preprocessing

        # apply optional apodization loss
        if apodization_alpha > 0 and apodization_factor > 0:
            nx = x_pred.shape[1]
            alpha = 2 * apodization_alpha / nx
            w = tukey(nx, alpha)
            w = np.outer(w, w).reshape((1, nx, nx, 1)).astype('float32')
            # Penalize non zero pixels near the border
            apo_loss = apodization_factor * tf.reduce_mean(tf.reduce_sum(((1. - w) * x_pred) ** 2, axis=[1, 2, 3]))
        else:
            w = 1.0
            apo_loss = 0.

        x_pred = x_pred * w

        # apply optional tv loss (penalize high frequencies features in the output)
        if tv_factor > 0:
            tv_loss = tv_factor * tf.image.total_variation(x_pred)
            # Smoothed Isotropic TV:
            # im_dx, im_dy = tf.image.image_gradients(x_pred)
            # tv_loss = tv_factor * tf.reduce_sum(tf.sqrt(im_dx**2 + im_dy**2 + 1e-6), axis=[1,2,3])
        else:
            tv_loss = 0.

        # compute loss in Fourier space (where covariance matrix is diagonal)
        x_true = tf.signal.rfft2d(image[..., 0])
        x_pred = tf.signal.rfft2d(x_pred[..., 0])

        chi_squared = 0.5 * tf.reduce_mean(tf.abs((x_true - x_pred))**2 / (tf.exp(ps)[..., 0] + 1e-8) / (2 * pi) ** 2, axis=[1, 2])
        return chi_squared + bottleneck_l2_cost + apo_loss + tv_loss
