import tensorflow as tf
from .layers.resnet_block import ResidualBlock
from .utils import get_activation


class ResnetEncoder(tf.keras.Model):
    def __init__(
            self,
            layers=7,
            res_blocks_in_layer=2,
            conv_layers_in_resblock=2,
            filter_scaling=2,
            filter_init=8,
            kernel_size_init=7,
            kernel_size=3,
            res_architecture="bare",
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
            mlp_bottleneck_neurons=16,
            batch_norm=False,
            dropout_rate=None,
            activation="relu"
    ):
        super(ResnetEncoder, self).__init__()
        self._num_layers = layers
        self.activation = get_activation(activation)
        self.res_blocks = []
        self.downsample_conv = []
        if isinstance(res_blocks_in_layer, list):
            assert len(res_blocks_in_layer) == layers
        else:
            res_blocks_in_layer = [res_blocks_in_layer] * layers
        self.mlp_bottleneck = tf.keras.layers.Dense(
            units=mlp_bottleneck_neurons,
            kernel_regularizer=tf.keras.regularizers.l2(l2=0.01)
        )
        for i in range(layers):
            self.downsample_conv.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(
                            filters=filter_init * int(filter_scaling ** (i + 1)),
                            kernel_size=kernel_size,
                            strides=2,
                            padding="same",
                            data_format="channels_last",
                            kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
                            bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp),
                        ),
                        tf.keras.layers.BatchNormalization() if batch_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x)),
                        self.activation
                    ]
                )
            )
            self.res_blocks.append(
                [
                    ResidualBlock(
                        filters=filter_init * int(filter_scaling ** i),
                        kernel_size=kernel_size,
                        conv_layers=conv_layers_in_resblock,
                        bias_reg_amp=bias_reg_amp,
                        kernel_reg_amp=kernel_reg_amp,
                        dropout_rate=dropout_rate,
                        architecture=res_architecture,
                        activation=activation
                    )
                    for j in range(res_blocks_in_layer[i])
                ]
            )
        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.input_layer = tf.keras.layers.Conv2D(
            filters=filter_init,
            kernel_size=kernel_size_init,
            padding="SAME",
            data_format="channels_last",
            bias_regularizer=tf.keras.regularizers.l2(l2=bias_reg_amp),
            kernel_regularizer=tf.keras.regularizers.l2(l2=kernel_reg_amp),
            activation=tf.keras.layers.ReLU()
        )

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        x = self.input_layer(x)
        for i in range(self._num_layers):
            for layers in self.res_blocks[i]:
                x = layers(x)
            x = self.downsample_conv[i](x)
        x = self.flatten(x)
        x = self.mlp_bottleneck(x)
        return x

    def call_with_skip_connections(self, x):
        """ Use in training autoencoder """
        skips = []
        x = self.input_layer(x)
        for i in range(self._num_layers):
            for layers in self.res_blocks[i]:
                x = layers(x)
            skips.append(tf.identity(x))
            x = self.downsample_conv[i](x)
        x = self.flatten(x)
        skips.append(tf.identity(x))
        x = self.mlp_bottleneck(x)
        return x, skips[::-1]