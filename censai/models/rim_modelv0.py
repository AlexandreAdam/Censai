import tensorflow as tf
from censai.models.layers import UnetDecodingLayer, UnetEncodingLayer
from .layers.conv_gru_component import ConvGRUBlock
from censai.definitions import DTYPE


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        common_params = {"padding": "same", "data_format": "channels_last"}

        self.L1

        self.bottleneck_gru = ConvGRUBlock(
            filters=2*bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

    def __call__(self, xt, states, grad):
        return self.call(xt, states, grad)

    def init_hidden_states(self, input_pixels, batch_size, constant=0.):
        pixels = input_pixels // self._strides ** (self._num_layers)
        return constant * tf.ones(shape=[batch_size, pixels, pixels, 2 * self._bottleneck_filters], dtype=DTYPE)
