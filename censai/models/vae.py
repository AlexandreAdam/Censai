import tensorflow as tf
from .decoder import Decoder
from .encoder import Encoder
from censai.definitions import DTYPE
from .utils import get_activation


class VAE(tf.keras.Model):
    def __init__(
            self,
            pixels=128,  # side length of the input image, used to compute shape of bottleneck mainly
            layers=7,
            conv_layers=2,
            filter_scaling=2,
            filters=8,
            kernel_size=3,
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            activation="bipolar_relu",
            dropout_rate=None,
            batch_norm=False,
            latent_size=16,
            strides=2,
            output_activation="softplus"
    ):
        super(VAE, self).__init__(dtype=DTYPE)
        output_activation = get_activation(output_activation)
        self.latent_size = latent_size
        self.encoder = Encoder(
            layers=layers,
            conv_layers=conv_layers,
            filter_scaling=filter_scaling,
            filters=filters,
            kernel_size=kernel_size,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            latent_size=latent_size,
            activation=activation,
            strides=strides
        )
        # compute size of mlp bottleneck from size of image and # of filters in the last encoding layer
        b_filters = filters * int(filter_scaling ** layers)
        pix = pixels//strides ** layers
        mlp_bottleneck = b_filters * pix**2
        self.decoder = Decoder(
            mlp_bottleneck=mlp_bottleneck,
            z_reshape_pix=pix,
            layers=layers,
            conv_layers=conv_layers,
            filter_scaling=filter_scaling,
            filters=filters,
            kernel_size=kernel_size,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation,
            strides=strides,
            output_activation=output_activation
        )

    def encode(self, x):
        batch_size = x.shape[0]
        mean, logvar = self.encoder(x)
        logvar = 0.5 * logvar
        epsilon = tf.random.normal([batch_size, self.latent_size], dtype=DTYPE)
        z = mean + tf.multiply(epsilon, tf.exp(logvar))
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        z, mean, logvar = self.encode(x)
        y = self.decode(z)
        return y

    def sample(self, batch_size=1):
        z = tf.random.normal([batch_size, self.latent_size], dtype=DTYPE)
        y = self.decode(z)
        return y

    def cost_function(self, x):
        z, mean, logvar = self.encode(x)
        y = self.decode(z)
        img_cost = tf.reduce_sum((y - x)**2, axis=(1, 2, 3))
        latent_cost = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=1)
        return img_cost + latent_cost

    def cost_training(self, x, l2_bottleneck):
        batch_size = x.shape[0]
        mean, logvar, pre_mlp_code = self.encoder.call_training(x)
        logvar = 0.5 * logvar
        epsilon = tf.random.normal([batch_size, self.latent_size], dtype=DTYPE)
        z = mean + tf.multiply(epsilon, tf.exp(logvar))
        y, bottleneck_l2_loss = self.decoder.call_training(z, pre_mlp_code, l2_bottleneck)
        reconstruction_loss = tf.reduce_sum((y - x)**2, axis=(1, 2, 3))
        kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=1)
        return reconstruction_loss, kl_loss, bottleneck_l2_loss