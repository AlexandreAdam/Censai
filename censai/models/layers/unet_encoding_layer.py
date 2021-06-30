import tensorflow as tf
from censai.models.utils import get_activation, summary_histograms
from censai.definitions import DTYPE


class UnetEncodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for downsampling. The output of the layer
    just before the downsampling is returned for the skip connection in the Unet
    """
    def __init__(
            self,
            kernel_size=3,
            downsampling_kernel_size=None,
            filters=32,
            conv_layers=2,
            activation="linear",
            name=None,
            strides=2,     # for final layer
            record=False,  # eventually record output of layer
            **kwargs
    ):
        super(UnetEncodingLayer, self).__init__(name=name, dtype=DTYPE)
        self.kernel_size = tuple([kernel_size]*2)
        if downsampling_kernel_size is None:
            self.downsampling_kernel_size = self.kernel_size
        else:
            self.downsampling_kernel_size = tuple([downsampling_kernel_size]*2)
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2)
        self.activation = get_activation(activation)
        self.conv_layers = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    activation=self.activation,
                    **kwargs
                )
            )
        self.downsampling_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.downsampling_kernel_size,
            strides=self.strides,
            activation=self.activation,
            **kwargs
        )

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x_down = self.downsampling_layer(x)
        return x, x_down
