from .conv_gru_plus_highway import ConvGRUPlusHighway
import tensorflow as tf


class ConvGRUPlusHighwayBlock(tf.keras.Model):
    """
    Abstraction for the recurrent block inside the RIM
    """
    def __init__(
            self,
            filters,
            kernel_size=5
    ):
        super(ConvGRUPlusHighwayBlock, self).__init__()
        kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            activation=tf.nn.tanh,
            padding='same',
            data_format="channels_last"
        )
        self.gru1 = ConvGRUPlusHighway(filters, kernel_size)
        self.gru2 = ConvGRUPlusHighway(filters, kernel_size)

    def call(self, inputs, state):
        ht_11, ht_12 = tf.split(state, 2, axis=3)
        gru_1_out  = self.gru1(inputs, ht_11)
        gru_1_outE = self.conv1(gru_1_out)
        gru_1_outE = gru_1_outE + gru_1_out  # skip connection
        gru_2_out  = self.gru2(gru_1_outE, ht_12)
        ht = tf.concat([gru_1_out, gru_2_out], axis=3)
        xt = gru_2_out
        return xt, ht