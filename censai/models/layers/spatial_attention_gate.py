import tensorflow as tf


class SpatialAttentionGate(tf.keras.layers.Layer):
    """
    Oktay, Schlemper, et al.
    Attention U-Net: Learning Where to Look for the Pancreas (2018),
    """
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(SpatialAttentionGate, self).__init__()
        # filters should be the number of filters in the skip connection
        self.c1_up = tf.keras.layers.Conv2D(filters, kernel_size, strides=(1,1), **kwargs)
        self.c1_skip = tf.keras.layers.Conv2D(filters, kernel_size, strides=(2,2), **kwargs)
        self.psi = tf.keras.layers.Conv2D(1, (1,1), strides=(1,1), **kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")

    def call(self, x_up, x_skip):
        x_up = self.c1_up(x_up)
        x_skip_down = self.c1_skip(x_skip)
        x_aligned = tf.nn.relu(tf.add(x_up, x_skip_down))
        psi = tf.nn.sigmoid(self.psi(x_aligned))
        psi = self.upsample(psi)
        out = psi * x_skip
        return out

