import tensorflow as tf


class ConcatELU(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatELU, self).__init__()

    def __call__(self, x):
        return tf.concat([tf.nn.elu(x), tf.nn.elu(-x)], axis=-1)


class GatedConv(tf.keras.layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(3,3),
            padding="SAME",
            data_format="channels_last",
        )
        self.concat_elu = ConcatELU()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=2 * filters,
            kernel_size=(1,1),
            padding="SAME",
            data_format="channels_last",
        )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.concat_elu(x)
        x = self.conv2(x)
        val, gate = tf.split(x, 2, axis=-1)
        return x + val * tf.nn.sigmoid(gate)