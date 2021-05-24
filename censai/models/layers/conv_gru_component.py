from .conv_gru import ConvGRU
import tensorflow as tf
from censai.models.utils import get_activation


class ConvGRUBlock(tf.keras.Model):
    """
    Abstraction for the recurrent block inside the RIM
    """
    def __init__(
            self,
            conv_filters,
            kernel_size=5,
            gru_filters=None,  # by default twice conv_filters
            activation="leaky_relu",
            **kwargs
    ):
        if gru_filters is None:
            gru_filters = 2 * conv_filters
        super(ConvGRUBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=conv_filters,
            kernel_size=kernel_size,
            strides=1,
            activation=get_activation(activation, **kwargs),
            padding='same')
        self.gru1 = ConvGRU(gru_filters)
        self.gru2 = ConvGRU(gru_filters)

    def call(self, inputs, state):
        ht_11, ht_12 = tf.split(state, 2, axis=3)
        gru_1_out  = self.gru1(inputs, ht_11)
        gru_1_outE = self.conv1(gru_1_out)
        gru_2_out  = self.gru2(gru_1_outE, ht_12)
        ht = tf.concat([gru_1_out, gru_2_out], axis=3)
        xt = gru_2_out
        return xt, ht