import tensorflow as tf
from .layers.conv_decoding_layer import ConvDecodingLayer


class Decoder(tf.keras.Model):
    def __init__(
            self,
            mlp_bottleneck,  # should match flattened dimension before mlp in encoder
            z_reshape_pix,   # should match dimension of encoder layer pre-mlp
            layers=7,
            conv_layers=2,
            filter_scaling=2,
            filters=8,
            kernel_size=3,
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            activation="relu",
            dropout_rate=None,
            batch_norm=False,
            bilinear=False
    ):
        super(Decoder, self).__init__()
        self._z_pix = z_reshape_pix
        self._z_filters = filters*(int(filter_scaling**layers))
        self._num_layers = layers
        self.conv_blocks = []
        for i in reversed(range(layers)):
            self.conv_blocks.append(
                ConvDecodingLayer(
                    filters=filters * int(filter_scaling ** i),
                    kernel_size=kernel_size,
                    conv_layers=conv_layers,
                    bias_reg_amp=bias_reg_amp,
                    kernel_reg_amp=kernel_reg_amp,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    bilinear=bilinear,
                    activation=activation
                )
            )
        self.mlp_bottleneck = tf.keras.layers.Dense(
            units=mlp_bottleneck,
            kernel_regularizer=tf.keras.regularizers.l2(l2=kernel_reg_amp)
        )
        self.output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding="SAME",
            activation="linear",
            kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp)
        )

    def __call__(self, z):
        return self.call(z)

    def call(self, z):
        z = self.mlp_bottleneck(z)
        batch_size = z.shape[0]
        x = tf.reshape(z, [batch_size, self._z_pix, self._z_pix, self._z_filters])
        for layer in self.conv_blocks:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def call_with_skip_connections(self, z, skips: list, skip_strength, l2):
        z = self.mlp_bottleneck(z)
        # add l2 cost for latent representation (we want identity map between this stage and pre-mlp stage of encoder)
        bottleneck_l2_cost = l2 * tf.reduce_mean((z - skips[0])**2, axis=1)
        batch_size = z.shape[0]
        x = tf.reshape(z, [batch_size, self._z_pix, self._z_pix, self._z_filters])
        for i, layer in enumerate(self.conv_blocks):
            x = layer.call_with_skip_connection(x, skip_strength * skips[i + 1])
        x = self.output_layer(x)
        return x, bottleneck_l2_cost
