import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE


class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            activation,
            strides,
            batch_norm,
            **kwargs
    ):
        super(DownsamplingLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )
        self.batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x))
        self.activation = activation

    def call(self, x, training=True):
        x = self.conv(x, training=training)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        return x


class UnetEncodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for downsampling. The output of the layer
    just before the downsampling is returned for the skip connection in the Unet
    """
    def __init__(
            self,
            filters=32,
            conv_layers=2,
            kernel_size=3,
            downsampling_kernel_size=None,
            downsampling_filters=None,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            name=None,
            strides=2,     # for final layer
            **kwargs
    ):
        super(UnetEncodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        if downsampling_kernel_size is None:
            self.downsampling_kernel_size = self.kernel_size
        else:
            self.downsampling_kernel_size = (downsampling_kernel_size,)*2 if isinstance(downsampling_kernel_size, int) else downsampling_kernel_size
        if downsampling_filters is None:
            downsampling_filters = filters
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = (strides,)*2 if isinstance(strides, int) else strides
        self.activation = get_activation(activation)

        self.conv_layers = []
        self.batch_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    **kwargs
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
        self.downsampling_layer = DownsamplingLayer(
            filters=downsampling_filters,
            kernel_size=self.downsampling_kernel_size,
            strides=self.strides,
            batch_norm=batch_norm,
            activation=self.activation,
            **kwargs
        )
        if dropout_rate is None:
            self.dropout = tf.identity
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x, training=True):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, training=training)
            x = self.batch_norms[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        x_down = self.downsampling_layer(x, training=training)
        return x, x_down
