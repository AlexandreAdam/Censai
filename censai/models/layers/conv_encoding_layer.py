import tensorflow as tf
from censai.models.utils import get_activation


class ConvEncodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for downsampling
    """
    def __init__(
            self,
            kernel_size=3,
            filters=32,
            conv_layers=2,
            activation="linear",
            name=None,
            strides=2,  # for final layer
            **common_params
    ):
        super(ConvEncodingLayer, self).__init__(name=name)
        self.kernel_size = tuple([kernel_size]*2)
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
                    **common_params
                )
            )
        self.downsample_layer = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            activation=self.activation,
            **common_params
        )

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.downsample_layer(x)
        return x
