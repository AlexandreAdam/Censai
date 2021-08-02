import tensorflow as tf
from censai.models.utils import get_activation


class ConvEncodingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size=3,
            downsampling_kernel_size=None,
            filters=32,
            conv_layers=2,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            name=None,
            strides=2,
            **common_params
    ):
        super(ConvEncodingLayer, self).__init__(name=name)
        if downsampling_kernel_size is None:
            self.downsampling_kernel_size = self.kernel_size
        else:
            self.downsampling_kernel_size = tuple([downsampling_kernel_size]*2)
        self.kernel_size = tuple([kernel_size]*2)
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2)
        self.activation = get_activation(activation)

        self.conv_layers = []
        self.batch_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    **common_params
                )
            )
            if batch_norm:
                self.batch_norms.append(
                    tf.keras.layers.BatchNormalization()
                )
            else:
                self.batch_norms.append(
                    tf.identity
                )
        self.downsampling_layer = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.downsampling_kernel_size,
                    strides=self.strides,
                    **common_params
                ),
                tf.keras.layers.BatchNormalization() if batch_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x)),
                self.activation
            ]
        )

        if dropout_rate is None:
            self.dropout = tf.identity
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.downsample_layer(x)
        return x

    def call_with_skip_connection(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x_down = self.downsampling_layer(x)
        return x, x_down
