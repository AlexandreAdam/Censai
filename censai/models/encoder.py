import tensorflow as tf
from .utils import get_activation
from .layers import UnetEncodingLayer


class Encoder(tf.keras.Model):
    def __init__(
            self,
            layers=7,
            conv_layers=2,
            filter_scaling=2,
            filters=8,
            kernel_size_init=7,
            kernel_size=3,
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            latent_size=16,
            batch_norm=False,
            activation="relu",
            dropout_rate=None
    ):
        super(Encoder, self).__init__()
        common_params = {"padding": "same",
                         "data_format": "channels_last",
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_reg_amp),
                         "bias_regularizer": tf.keras.regularizers.L2(l2=bias_reg_amp)}
        self._num_layers = layers
        self.activation = get_activation(activation)
        self.conv_layers = []
        self.bottleneck_layer = tf.keras.layers.Dense(
            units=latent_size,
            kernel_regularizer=tf.keras.regularizers.l2(l2=kernel_reg_amp)
        )
        for i in range(layers):
            self.conv_layers.append(
                UnetEncodingLayer(
                    kernel_size=kernel_size,
                    filters=filters * int(filter_scaling ** i),
                    downsampling_filters=filters * int(filter_scaling ** (i+1)),
                    conv_layers=conv_layers,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                    **common_params
                )
            )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.input_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size_init,
            activation=tf.keras.layers.ReLU(),
            **common_params
        )

    def __call__(self, x):
        return self.call(x)

    def call(self, x, training=True):
        x = self.input_layer(x, training=training)
        for i, layer in enumerate(self.conv_layers):
            _, x = layer(x, training=training)
        x = self.flatten(x)
        x = self.bottleneck_layer(x, training=training)
        return x

    def call_with_skip_connections(self, x):
        skips = []
        x = self.input_layer(x)
        for i, layer in enumerate(self.conv_layers):
            skip, x = layer(x)
            skips.append(skip)
        x = self.flatten(x)
        skips.append(tf.identity(x))
        x = self.bottleneck_layer(x)
        return x, skips[::-1]
