import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from tensorflow_addons.layers import GroupNormalization


class DownsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            **kwargs
    ):
        super(DownsamplingLayer, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )

    def call(self, x):
        x = self.conv(x)
        return x


class ResUnetAtrousEncodingLayer(tf.keras.layers.Layer):
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
            group_norm=False,
            groups=1,
            dropout_rate=None,
            name=None,
            strides=2,
            dilation_rates=[1],
            **common_params
    ):
        super(ResUnetAtrousEncodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = tuple([kernel_size]*2)
        if downsampling_kernel_size is None:
            self.downsampling_kernel_size = self.kernel_size
        else:
            self.downsampling_kernel_size = tuple([downsampling_kernel_size]*2)
        if downsampling_filters is None:
            downsampling_filters = filters
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2)
        self.activation = get_activation(activation)

        self.groups = []
        for d in dilation_rates:
            group = []
            for i in range(self.num_conv_layers):
                if group_norm:
                    group.append(GroupNormalization(groups))
                group.append(activation)
                if dropout_rate is not None:
                    group.append(tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last"))
                group.append(
                    tf.keras.layers.Conv2DTranspose(
                        filters=self.filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=d,
                        **common_params
                    )
                )
            self.groups.append(group)

        self.downsampling_layer = DownsamplingLayer(
            filters=downsampling_filters,
            kernel_size=self.downsampling_kernel_size,
            strides=self.strides,
            **common_params
        )

    def call(self, x):
        out = []
        for group in self.groups:
            z = tf.identity(x)
            for layer in group:
                z = layer(z)
            out.append(tf.identity(z))
        y = tf.add_n(out)
        x = y + x  # resnet connection
        x_down = self.downsampling_layer(x)
        return x, x_down
