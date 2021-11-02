import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from .blurpool import BlurPool2D


class ShuffleUnetDecodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for upsampling. Skip connection
    occurs before this layer.
    """
    def __init__(
            self,
            kernel_size=3,
            filters=32,
            conv_layers=2,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            blur=False,
            blur_kernel_size=3,
            name=None,
            **common_params
    ):
        super(ShuffleUnetDecodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.activation = get_activation(activation)

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
        self.upsampling_layer = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, bloc_size=2, data_format="NHWC"))
        if blur:
            self.blur_layer = BlurPool2D(kernel_size=blur_kernel_size)
        else:
            self.blur_layer = tf.identity
        if dropout_rate is None:
            self.dropout = tf.keras.layers.Lambda(lambda x, training=True: x)
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x, c_i, training=True):  # c_i is the skip connection
        x = self.upsampling_layer(x)
        x = self.blur_layer(x)
        x = tf.concat([x, c_i], axis=-1)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, training=training)
            x = self.batch_norms[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        return x
