import tensorflow as tf
from .decoder import Decoder
from .resnet_encoder import ResnetEncoder
from censai.definitions import DTYPE


class ResnetVAE(tf.keras.Model):
    def __init__(
            self,
            pixels=128,  # side length of the input image, used to compute shape of bottleneck mainly
            layers=7,
            res_blocks_in_layer=2,
            conv_layers_per_block=2,
            filter_scaling=2,
            filters=8,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            activation="bipolar_relu",
            dropout_rate=None,
            batch_norm=False,
            latent_size=16
    ):
        super(ResnetVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = ResnetEncoder(
            layers=layers,
            res_blocks_in_layer=res_blocks_in_layer,
            conv_layers_in_resblock=conv_layers_per_block,
            filter_scaling=filter_scaling,
            filter_init=filters,
            kernel_size=kernel_size,
            res_architecture=res_architecture,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            mlp_bottleneck_neurons=latent_size,
            activation=activation
        )
        # compute size of mlp bottleneck from size of image and # of filters in the last encoding layer
        b_filters = filters * (int(filter_scaling ** layers))
        pix = pixels//2 ** layers
        mlp_bottleneck = b_filters * pix**2
        self.decoder = Decoder(
            mlp_bottleneck=mlp_bottleneck,
            z_reshape_pix=pix,
            layers=layers,
            conv_layers=conv_layers_per_block,
            filter_scaling=filter_scaling,
            filters=filters,
            kernel_size=kernel_size,
            kernel_reg_amp=kernel_reg_amp,
            bias_reg_amp=bias_reg_amp,
            dropout_rate=dropout_rate,
            batch_norm=batch_norm,
            activation=activation
        )

    def encode(self, x):
        batch_size = x.shape[0]
        distribution_parameters = self.encoder(x)
        mean, logvar = tf.split(distribution_parameters, 2, axis=1)
        logvar = 0.5 * logvar
        epsilon = tf.random.normal([batch_size, self.latent_size//2], dtype=DTYPE)
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

    def cost_function(self, x):
        z, mean, logvar = self.encode(x)
        y = self.decode(z)
        img_cost = tf.reduce_sum((y - x)**2, axis=(1, 2, 3))
        latent_cost = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=1)
        return img_cost + latent_cost

    def cost_function_training(self, x, skip_strength, l2_bottleneck):
        batch_size = x.shape[0]
        distribution_parameters, skip_connections = self.encoder.call_with_skip_connections(x)
        mean, logvar = tf.split(distribution_parameters, 2, axis=1)
        logvar = 0.5 * logvar
        epsilon = tf.random.normal([batch_size, self.latent_size // 2], dtype=DTYPE)
        z = mean + tf.multiply(epsilon, tf.exp(logvar))
        y, bottleneck_l2_loss = self.decoder.call_with_skip_connections(z, skip_connections, skip_strength, l2_bottleneck)
        reconstruction_loss = tf.reduce_sum((y - x)**2, axis=(1, 2, 3))
        kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=1)
        return reconstruction_loss, kl_loss, bottleneck_l2_loss
