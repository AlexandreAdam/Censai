import tensorflow as tf
from .layers.resnet_block import ResidualBlock


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
        self.mlp_bottleneck = tf.keras.layers.Dense(
            units=mlp_bottleneck_neurons,
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)
        )
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
        self.mlp_bottleneck = tf.keras.layers.Dense(
            units=mlp_bottleneck,
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)
        )
        self.output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding="SAME",
            activation="linear",
            kernel_initializer=kernel_initializer,
        )

    def __call__(self, z):
        return self.call(z)

    def call(self, z):
        z = self.mlp_bottleneck(z)
        batch_size, _ = z.shape
        x = tf.reshape(z, [batch_size, self._z_pix, self._z_pix, self._z_filters])
        for i in range(self._num_layers):
            x = self.upsample_conv[i](x)
            x = self.res_blocks[i](x)
        x = self.output_layer(x)
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
        x = self.output_layer(x)
        return x, bottleneck_l2_cost


class ResnetAutoencoder(tf.keras.Model):
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
            alpha=0.1,
            resblock_dropout_rate=None,
            kernel_initializer="he_uniform",
            latent_size=16,
            **kwargs
    ):
        super(ResnetAutoencoder, self).__init__()
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

