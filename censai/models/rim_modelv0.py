import tensorflow as tf
from .layers import ConvGRU
from censai.definitions import DTYPE


class ModelMorningstar(tf.keras.Model):
    def __init__(self, filters=32):
        super(ModelMorningstar, self).__init__()

        common_params = {"padding": "same", "data_format": "channels_last"}
        self.filters = filters
        self.input_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=(11, 11), strides=4, **common_params)
        self.gru1 = ConvGRU(filters=filters, kernel_size=(11, 11))
        self.middle_layer = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(11, 11), strides=4, **common_params)
        self.gru2 = ConvGRU(filters=filters, kernel_size=(11, 11))
        self.output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), **common_params)

    def __call__(self, xt, states, grad):
        return self.call(xt, states, grad)

    def call(self, xt, states, grad):
        dx = tf.concat([xt, grad], axis=3)
        ht11, ht12 = states
        dx = tf.nn.tanh(self.input_layer(dx))
        ht21, _ = self.gru1(dx, ht11)
        ht21_features = tf.nn.tanh(self.middle_layer(ht21))
        ht22, _ = self.gru2(ht21_features, ht12)
        dx = self.output_layer(ht22)
        new_states = tf.concat([ht21, ht22], axis=3)
        xt1 = xt + dx
        return xt1, new_states

    def init_hidden_states(self, pixels, batch_size):
        ht1 = tf.zeros(shape=[batch_size, pixels//4, pixels//4, self.filters], dtype=DTYPE)
        ht2 = tf.zeros(shape=[batch_size, pixels, pixels, self.filters], dtype=DTYPE)
        return ht1, ht2
