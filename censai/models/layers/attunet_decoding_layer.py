import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from .spatial_attention_gate import SpatialAttentionGate


class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            batch_norm,
            activation,
            **kwargs
    ):
        super(UpsamplingLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )
        self.batch_norm = tf.keras.layers.BatchNormalization() if batch_norm else tf.keras.layers.Lambda(lambda x, training=True: x)
        self.activation = activation

    def call(self, x, training=True):
        x = self.conv(x, training=training)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        return x


class AttUnetDecodingLayer(tf.keras.layers.Layer):
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
            batch_norm=False,
            dropout_rate=None,
            name=None,
            strides=2,       # for final layer
            bilinear=False,  # whether to use bilinear upsampling or vanilla half strid convolution
            **common_params
    ):
        super(AttUnetDecodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        if upsampling_kernel_size is None:
            self.upsampling_kernel_size = self.kernel_size
        else:
            self.upsampling_kernel_size = (upsampling_kernel_size,)*2 if isinstance(upsampling_kernel_size, int) else upsampling_kernel_size
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2) if isinstance(strides, int) else strides
        self.activation = get_activation(activation)
        self.att_gate = SpatialAttentionGate(filters, kernel_size, **common_params)

        self.conv_layers = []
        self.batch_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    **common_params
                )
            )
            if batch_norm:
                self.batch_norms.append(
                    tf.keras.layers.BatchNormalization()
                )
            else:
                self.batch_norms.append(
                    tf.keras.layers.Lambda(lambda x, training=True: x)
                )

        self.upsampling_layer = UpsamplingLayer(
            filters=filters,
            kernel_size=self.upsampling_kernel_size,
            strides=self.strides,
            batch_norm=batch_norm,
            activation=self.activation,
            **common_params

        )
        if dropout_rate is None:
            self.dropout = tf.keras.layers.Lambda(lambda x, training=True: x)
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x, c_i, training=True):  # c_i is the skip connection
        c_i = self.att_gate(x_up=x, x_skip=c_i)
        x = self.upsampling_layer(x, training=training)
        x = tf.concat([x, c_i], axis=-1)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, training=training)
            x = self.batch_norms[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x
