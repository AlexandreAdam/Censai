import tensorflow as tf
from censai.definitions import DTYPE


class ConvGRUPlus(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=5, **kwargs):
        super(ConvGRUPlus, self).__init__(dtype=DTYPE, **kwargs)
        self.filters = filters
        self.kernel_size = (kernel_size,)*2 if isinstance(kernel_size, int) else kernel_size

    def build(self, input_shape):
        self.w_z = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False,
        )
        self.u_z = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False
        )
        self.bias_z = tf.Variable(tf.zeros(shape=input_shape[1:], dtype=DTYPE))

        self.w_r = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False
        )
        self.u_r = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False
        )
        self.bias_r = tf.Variable(tf.zeros(shape=input_shape[1:], dtype=DTYPE))

        self.w_h = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False
        )
        self.u_h = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=(1, 1),
            padding='same',
            data_format="channels_last",
            use_bias=False
        )
        self.bias_h = tf.Variable(tf.zeros(shape=input_shape[1:], dtype=DTYPE))

    def call(self, x, ht):
        """
        Compute the new state tensor h_{t+1}.
        """
        z = tf.nn.sigmoid(self.w_z(x) + self.u_z(ht) + self.bias_z)  # update gate
        r = tf.nn.sigmoid(self.w_r(x) + self.u_r(ht) + self.bias_r)  # reset gate
        h_tilde = tf.nn.tanh(self.w_h(x) + self.u_h(r * ht) + self.bias_h)  # candidate activation
        new_state = (1 - z)*ht + z*h_tilde
        return new_state, new_state  # h_{t+1}
