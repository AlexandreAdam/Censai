from .conv_gru import ConvGRU
import tensorflow as tf


class ConvGRUBlock(tf.keras.Model):
    """
    Abstraction for the recurrent block inside the RIM
    """
    def __init__(
            self,
            num_cell_features,
    ):
        super(ConvGRUBlock, self).__init__()
        self.kernel_size = 5
        self.num_gru_features = num_cell_features/2
        self.conv1 = tf.keras.layers.Conv2D(filters=self.num_gru_features, kernel_size=self.kernel_size, strides=1, activation='relu', padding='same')
        self.gru1 = ConvGRU(self.num_gru_features)
        self.gru2 = ConvGRU(self.num_gru_features)

    def call(self, inputs, state):
        ht_11, ht_12 = tf.split(state, 2, axis=3)
        gru_1_out  = self.gru1(inputs, ht_11)
        gru_1_outE = self.conv1(gru_1_out)
        gru_2_out  = self.gru2(gru_1_outE, ht_12)
        ht = tf.concat([gru_1_out, gru_2_out], axis=3)
        xt = gru_2_out
        return xt, ht