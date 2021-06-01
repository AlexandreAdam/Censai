import tensorflow as tf
from .layers.resnet_block import ResidualBlock
from numpy import pi
from censai.galflow import convolve
from scipy.signal.windows import tukey
import numpy as np


class Encoder(tf.keras.Model):
    def __init__(
            self,
            res_layers=7,
            conv_layers_in_resblock=2,
            filter_scaling=2,
            filter_init=8,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            alpha=0.04,
            resblock_dropout_rate=None,
            kernel_initializer="he_uniform",
            mlp_bottleneck_neurons=16,
            **kwargs
    ):
        super(Encoder, self).__init__()
        self._num_layers = res_layers
        self.res_blocks = []
        self.downsample_conv = []
        self.mlp_bottleneck = tf.keras.layers.Dense(mlp_bottleneck_neurons)
        for i in range(res_layers):
            self.downsample_conv.append(
                tf.keras.layers.Conv2D(
                    filters=filter_init * int(filter_scaling ** i),
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    data_format="channels_last",
                    kernel_initializer=kernel_initializer,
                    activation=tf.keras.layers.LeakyReLU(alpha),
                    kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
                    bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp),
                )
            )
            self.res_blocks.append(
                ResidualBlock(
                    filters=filter_init * int(filter_scaling ** i),
                    kernel_size=kernel_size,
                    conv_layers=conv_layers_in_resblock,
                    bias_reg_amp=bias_reg_amp,
                    kernel_reg_amp=kernel_reg_amp,
                    alpha=alpha,
                    dropout_rate=resblock_dropout_rate,
                    architecture=res_architecture,
                    **kwargs
                )
            )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        for i in range(self._num_layers):
            x = self.downsample_conv[i](x)
            x = self.res_blocks[i](x)
        x = tf.keras.layers.Flatten(data_format="channels_last")(x)
        x = self.mlp_bottleneck(x)
        return x

    def call_with_skip_connections(self, x):
        """ Use in training autoencoder """
        skips = []
        for i in range(self._num_layers):
            x = self.downsample_conv[i](x)
            x = self.res_blocks[i](x)
            skips.append(tf.identity(x))
        x = self.flatten(x)
        skips.append(tf.identity(x))  # recorded for l2 loss of bottleneck. In reality, its just a reshaped copy of previous skip
        x = self.mlp_bottleneck(x)
        return x, skips[::-1]


class Decoder(tf.keras.Model):
    def __init__(
            self,
            mlp_bottleneck,  # should match flattened dimension before mlp in encoder
            z_reshape_pix,   # should match dimension of encoder layer pre-mlp
            res_layers=7,
            conv_layers_in_resblock=2,
            filter_scaling=2,
            filter_init=8,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            alpha=0.04,
            resblock_dropout_rate=None,
            kernel_initializer="he_uniform",
            **kwargs
    ):
        super(Decoder, self).__init__()
        self._z_pix = z_reshape_pix
        self._z_filters = filter_init*(int(filter_scaling**(res_layers-1)))
        self._num_layers = res_layers
        self.res_blocks = []
        self.upsample_conv = []
        for i in reversed(range(res_layers)):
            self.upsample_conv.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=(filter_init * max(1, int(filter_scaling ** (i - 1)))) ** (0 if i==0 else 1),
                    kernel_size=kernel_size,
                    strides=2,
                    padding="same",
                    data_format="channels_last",
                    kernel_initializer=kernel_initializer,
                    activation=tf.keras.layers.LeakyReLU(alpha),
                    kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
                    bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp),
                )
            )
            self.res_blocks.append(
                ResidualBlock(
                    filters=(filter_init * max(1, int(filter_scaling ** (i - 1)))) ** (0 if i==0 else 1),
                    kernel_size=kernel_size,
                    conv_layers=conv_layers_in_resblock,
                    bias_reg_amp=bias_reg_amp,
                    kernel_reg_amp=kernel_reg_amp,
                    alpha=alpha,
                    dropout_rate=resblock_dropout_rate,
                    architecture=res_architecture,
                    **kwargs
                )
            )
        self.mlp_bottleneck = tf.keras.layers.Dense(mlp_bottleneck)

    def __call__(self, z):
        return self.call(z)

    def call(self, z):
        z = self.mlp_bottleneck(z)
        batch_size, _ = z.shape
        x = tf.reshape(z, [batch_size, self._z_pix, self._z_pix, self._z_filters])
        for i in range(self._num_layers):
            x = self.upsample_conv[i](x)
            x = self.res_blocks[i](x)
        return x

    def call_with_skip_connections(self, z, skips: list, skip_strength, l2):
        z = self.mlp_bottleneck(z)
        # add l2 cost for latent representation (we want identity map between this stage and pre-mlp stage of encoder)
        bottleneck_l2_cost = l2 * tf.reduce_mean((z - skips[0])**2, axis=1)
        batch_size, _ = z.shape
        x = tf.reshape(z, [batch_size, self._z_pix, self._z_pix, self._z_filters])
        for i in range(self._num_layers):
            x = tf.add(x, skip_strength * skips[i+1])
            x = self.upsample_conv[i](x)
            x = self.res_blocks[i](x)
        return x, bottleneck_l2_cost


class Autoencoder(tf.keras.Model):
    def __init__(
            self,
            pixels=128,  # side length of the input image, used to compute shape of bottleneck mainly
            res_layers=7,
            conv_layers_in_resblock=2,
            filter_scaling=2,
            filter_init=8,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            alpha=0.04,
            resblock_dropout_rate=None,
            kernel_initializer="he_uniform",
            latent_size=16,
            image_floor=1e-8,
            **kwargs
    ):
        super(Autoencoder, self).__init__()
        self.image_floor = image_floor
        self.encoder = Encoder(
            res_layers=res_layers,
            conv_layers_in_resblock=conv_layers_in_resblock,
            filter_scaling=filter_scaling,
            filter_init=filter_init,
            kernel_size=kernel_size,
            res_architecture=res_architecture,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            alpha=alpha,
            resblock_dropout_rate=resblock_dropout_rate,
            kernel_initializer=kernel_initializer,
            mlp_bottleneck_neurons=latent_size,
            **kwargs
        )
        # compute size of mlp bottleneck from size of image and # of filters in the last encoding layer
        filters = filter_init*(int(filter_scaling**(res_layers-1)))
        pix = pixels//2**(res_layers)
        mlp_bottleneck = filters * pix**2
        self.decoder = Decoder(
            mlp_bottleneck=mlp_bottleneck,
            z_reshape_pix=pix,
            res_layers=res_layers,
            conv_layers_in_resblock=conv_layers_in_resblock,
            filter_scaling=filter_scaling,
            filter_init=filter_init,
            kernel_size=kernel_size,
            res_architecture=res_architecture,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            alpha=alpha,
            resblock_dropout_rate=resblock_dropout_rate,
            kernel_initializer=kernel_initializer,
            **kwargs
        )

    def link_function(self, x):
        return tf.math.log(x + self.image_floor)

    @staticmethod
    def inverse_link_function(eta):
        return tf.math.exp(eta)

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
        x = self.link_function(x)
        psf_image = self.link_function(psf_image)
        x = tf.concat([x, psf_image], axis=-1)
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x, psf):
        return self.call(x, psf)

    def call(self, x, psf_image):
        x = self.link_function(x)
        psf_image = self.link_function(psf_image)
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
        x_pred = self.inverse_link_function(x_pred)
        x_pred = convolve(x_pred, tf.cast(psf[..., 0], tf.complex64), zero_padding_factor=1)

        x = tf.signal.rfft2d(x[..., 0])
        x_pred = tf.signal.rfft2d(x_pred[..., 0])

        chi_squared = 0.5 * tf.reduce_mean(tf.abs((x - x_pred)**2 / tf.complex(tf.exp(ps) + 1e-8, 0.) / (2 * pi) ** 2), axis=[1, 2])
        return chi_squared

    def training_cost_function(self, x, psf, ps, skip_strength, l2_bottleneck, apodization_alpha, apodization_factor, tv_factor):
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
        input_shape = x.shape
        psf_image = tf.signal.irfft2d(tf.cast(psf[..., 0], tf.complex64))[..., tf.newaxis]
        # Roll the image to undo the fftshift, assuming x1 zero padding and x2 subsampling
        psf_image = tf.roll(psf_image, shift=[input_shape[1], input_shape[2]], axis=[1, 2])
        psf_image = tf.image.resize_with_crop_or_pad(psf_image, input_shape[1], input_shape[2])
        x = self.link_function(x)
        psf_image = self.link_function(psf_image)
        x = tf.concat([x, psf_image], axis=-1)

        z, skips = self.encoder.call_with_skip_connections(x)
        x_pred, bottleneck_l2_cost = self.decoder.call_with_skip_connections(z, skips, skip_strength, l2_bottleneck)
        x_pred = self.inverse_link_function(x_pred)
        x_pred = convolve(x_pred, tf.cast(psf[..., 0], tf.complex64), zero_padding_factor=1) # we already padded psf with noise in data preprocessing

        # We apply an optional apodization of the output before taking the
        if apodization_alpha > 0 and apodization_factor > 0:
            nx = x_pred.shape[1]
            alpha = 2 * apodization_alpha / nx
            # Create a tukey window
            w = tukey(nx, alpha)
            w = np.outer(w, w).reshape((1, nx, nx, 1)).astype('float32')
            # And penalize non zero things at the border
            apo_loss = apodization_factor * tf.reduce_mean(tf.reduce_sum(((1. - w) * x_pred) ** 2, axis=[1, 2, 3]))
        else:  # rectangular window
            w = 1.0
            apo_loss = 0.

        # We apply the window
        x_pred = x_pred * w

        # apply tv loss
        if tv_factor > 0:
            tv_loss = tv_factor * tf.image.total_variation(x_pred)
            # Smoothed Isotropic TV:
            # im_dx, im_dy = tf.image.image_gradients(x_pred)
            # tv_loss = tv_factor * tf.reduce_sum(tf.sqrt(im_dx**2 + im_dy**2 + 1e-6), axis=[1,2,3])
        else:
            tv_loss = 0.

        x = tf.signal.rfft2d(x[..., 0])
        x_pred = tf.signal.rfft2d(x_pred[..., 0])

        # added a safety net in the division, even if tfrecords were generated to ensure
        chi_squared = 0.5 * tf.reduce_mean(tf.abs((x - x_pred)**2 / tf.complex(tf.exp(ps)[..., 0] + 1e-8, 0.) / (2 * pi) ** 2), axis=[1, 2])
        return chi_squared + bottleneck_l2_cost + apo_loss + tv_loss
