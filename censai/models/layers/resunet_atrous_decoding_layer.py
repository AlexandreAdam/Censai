import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from tensorflow_addons.layers import GroupNormalization


class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            group_norm,
            groups=1, # default is equivalent to LayerNormalisation
            **kwargs
    ):
        super(UpsamplingLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )
        self.group_norm = GroupNormalization(groups) if group_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x))

    def call(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        return x


class ResUnetAtrousDecodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for upsampling. Skip connection
    occurs before this layer.
    """
    def __init__(
            self,
            kernel_size=3,
            upsampling_kernel_size=None,
            filters=32,
            conv_layers=2,
            activation="linear",
            group_norm=False,
            groups=1,
            dropout_rate=None,
            name=None,
            strides=2,       # for final layer
            dilation_rates=[1],
            **common_params
    ):
        super(ResUnetAtrousDecodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = tuple([kernel_size]*2)
        if upsampling_kernel_size is None:
            self.upsampling_kernel_size = self.kernel_size
        else:
            self.upsampling_kernel_size = tuple([upsampling_kernel_size]*2)
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

        self.upsampling_layer = UpsamplingLayer(
            filters=self.filters,
            kernel_size=self.upsampling_kernel_size,
            strides=self.strides,
            group_norm=group_norm,
            groups=groups,
            **common_params
            )
        if dropout_rate is None:
            self.dropout = tf.identity

        self.combine = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            **common_params
        )

    def call(self, x, c_i):  # c_i is the skip connection
        x = self.upsampling_layer(x)
        x = self.combine(tf.concat([x, self.activation(c_i)], axis=-1))
        out = []
        for group in self.groups:
            z = tf.identity(x)
            for layer in group:
                z = layer(z)
            out.append(tf.identity(z))
        z = tf.add_n(out)
        return x + z
