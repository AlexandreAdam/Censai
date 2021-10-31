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
            activation,
            groups=1,
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
        self.activation = activation

    def call(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = self.activation(x)
        return x


class ResUnetDecodingLayer(tf.keras.layers.Layer):
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
            group_norm=True,
            groups=1,
            dropout_rate=None,
            name=None,
            strides=2,       # for final layer
            bilinear=False,  # whether to use bilinear upsampling or vanilla half stride convolution
            **common_params
    ):
        super(ResUnetDecodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = tuple([kernel_size]*2) if isinstance(kernel_size, int) else kernel_size
        if upsampling_kernel_size is None:
            self.upsampling_kernel_size = self.kernel_size
        else:
            self.upsampling_kernel_size = (upsampling_kernel_size,)*2 if isinstance(upsampling_kernel_size, int) else upsampling_kernel_size
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2) if isinstance(strides, int) else strides
        self.activation = get_activation(activation)

        self.conv_layers = []
        self.group_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    **common_params
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
        if bilinear:
            self.upsampling_layer = tf.keras.layers.UpSampling2D(size=self.strides, interpolation="bilinear")
        else:
            self.upsampling_layer = UpsamplingLayer(
                filters=self.filters,
                kernel_size=self.upsampling_kernel_size,
                strides=self.strides,
                group_norm=group_norm,
                activation=self.activation,
                **common_params

            )
        if dropout_rate is None:
            self.dropout = tf.keras.layers.Lambda(lambda x, training=True: x)
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

        self.combine = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1,)*2,
            **common_params
        )

    def call(self, y, c_i):  # c_i is the skip connection
        y = self.upsampling_layer(y)
        y = self.combine(tf.concat([y, c_i], axis=-1))
        x = tf.identity(y)
        for i, layer in enumerate(self.conv_layers):
            x = self.group_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            x = layer(x)
        return y + x
