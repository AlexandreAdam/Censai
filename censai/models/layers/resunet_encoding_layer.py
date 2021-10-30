import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from tensorflow_addons.layers import GroupNormalization


class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            activation,
            strides,
            group_norm,
            groups=1,
            **kwargs
    ):
        super(DownsamplingLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )
        self.group_norm = GroupNormalization(groups) if group_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x))
        self.activation = activation

    def call(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.activation(x)
        return x


class ResUnetEncodingLayer(tf.keras.layers.Layer):
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
            group_norm=True,
            groups=1,
            dropout_rate=None,
            name=None,
            strides=2,     # for final layer
            **kwargs
    ):
        super(ResUnetEncodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = tuple([kernel_size]*2) if isinstance(kernel_size, int) else kernel_size
        if downsampling_kernel_size is None:
            self.downsampling_kernel_size = self.kernel_size
        else:
            self.downsampling_kernel_size = tuple([downsampling_kernel_size]*2)
        if downsampling_filters is None:
            downsampling_filters = filters
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2) if isinstance(strides, int) else strides
        self.activation = get_activation(activation)

        self.conv_layers = []
        self.group_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    **kwargs
                )
            )
            if group_norm:
                self.group_norms.append(
                    GroupNormalization(groups)
                )
            else:
                self.group_norms.append(
                    tf.keras.layers.Lambda(lambda x, training=True: x)
                )
        self.downsampling_layer = DownsamplingLayer(
            filters=downsampling_filters,
            kernel_size=self.downsampling_kernel_size,
            strides=self.strides,
            group_norm=group_norm,
            activation=self.activation,
            **kwargs
        )
        if dropout_rate is None:
            self.dropout = tf.keras.layers.Lambda(lambda x, training=True: x)
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x):
        y = tf.identity(x)
        for i, layer in enumerate(self.conv_layers):
            x = self.group_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = layer(x)
        x = y + x  # resnet connection
        x_down = self.downsampling_layer(x)
        return x, x_down
