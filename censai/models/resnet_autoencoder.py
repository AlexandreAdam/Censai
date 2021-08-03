import tensorflow as tf
from .decoder import Decoder
from .resnet_encoder import ResnetEncoder


class ResnetAutoencoder(tf.keras.Model):
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
        super(ResnetAutoencoder, self).__init__()
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        return self.decoder(self.encoder(x))

    def cost_function(self, x):
        y = self.call(x)
        loss = tf.reduce_mean(tf.square(y - x))
        return loss

