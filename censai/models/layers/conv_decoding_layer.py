import tensorflow as tf
from censai.models.utils import get_activation


class ConvDecodingLayer(tf.keras.layers.Layer):
    """
    Abstraction for n convolutional layers and a strided convolution for downsampling
    """
    def __init__(
            self,
            kernel_size=3,
            upsampling_kernel_size=None,
            filters=32,
            conv_layers=2,
            activation="linear",
            name=None,
            strides=2,
            bilinear=False,
            **common_params
    ):
        super(ConvDecodingLayer, self).__init__(name=name)
        if upsampling_kernel_size is None:
            self.upsampling_kernel_size = self.kernel_size
        else:
            self.upsampling_kernel_size = tuple([upsampling_kernel_size]*2)
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
        if bilinear:
            self.upsampling_layer = tf.keras.layers.UpSampling2D(size=self.strides, interpolation="bilinear")
        else:
            self.upsampling_layer = tf.keras.layers.Conv2DTranspose(
                filters=self.filters,
                kernel_size=self.upsampling_kernel_size,
                strides=self.strides,
                activation=self.activation,
                **common_params
            )

    def call(self, x):
        x = self.upsampling_layer(x)
        for layer in self.conv_layers:
            x = layer(x)
        return x
