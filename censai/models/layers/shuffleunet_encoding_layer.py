import tensorflow as tf
from censai.models.utils import get_activation
from censai.definitions import DTYPE
from .blurpool import BlurPool2D


class ShuffleUnetEncodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for downsampling. The output of the layer
    just before the downsampling is returned for the skip connection in the Unet
    """
    def __init__(
            self,
            filters=32,
            downsample_filters=None,
            conv_layers=2,
            kernel_size=3,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            blurpool_kernel_size=3,
            block_size=2, # equivalent to stride, used when blurpool is False
            name=None,
            **kwargs
    ):
        super(ShuffleUnetEncodingLayer, self).__init__(name=name, dtype=DTYPE)
        assert blurpool_kernel_size in [3, 5]
        self.kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.activation = get_activation(activation)
        if downsample_filters is None:
            downsample_filters = filters

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
                    tf.keras.layers.Lambda(lambda x, training=True: x)
                )
        self.fully_connected = tf.keras.layers.Conv2D(filters=downsample_filters, kernel_size=(1,1), **kwargs)
        self.downsampling_layer = tf.keras.layers.Lambda(lambda x: tf.nn.space_to_depth(x, block_size=block_size, data_format="NHWC"))
        self.blur_layer = BlurPool2D(pool_size=1, kernel_size=blurpool_kernel_size)
        if dropout_rate is None:
            self.dropout = tf.keras.layers.Lambda(lambda x, training=True: x)
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x, training=True):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x, training=training)
            x = self.batch_norms[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        x_down = self.downsampling_layer(x)
        x_down = self.blur_layer(x_down)
        x_down = self.fully_connected(x_down)
        x_down = self.activation(x_down)
        return x, x_down
