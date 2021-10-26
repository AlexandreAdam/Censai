import tensorflow as tf
import tensorflow_probability as tfp
from censai.definitions import DTYPE

tfd = tfp.distributions


class CouplingLayer(tf.keras.layers.Layer):
    def __init__(self, network, mask, channels_in):
        super(CouplingLayer, self).__init__()
        self.network = network
        self.scaling_factor = tf.constant(channels_in, DTYPE)[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        self.mask = tf.constant(mask, dtype=DTYPE)

    def __call__(self, z, ldj, reverse=False):
        return self.call(z, ldj, reverse)

    def call(self, z, ldj, reverse=False):
        z_in = z * self.mask
        nn_out = self.network(z_in)
        s, t = tf.split(nn_out, 2, axis=-1)

        # stabilize scaling output
        s_fac = tf.exp(self.scaling_factor)
        s = tf.nn.tanh(s / s_fac) * s_fac

        # mask outputs
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # affine transform
        if reverse:
            z = (z * tf.exp(-s)) - t
            ldj = ldj - tf.reduce_sum(s, axis=(1, 2, 3), keepdims=True)
        else:
            z = (z + t) * tf.exp(s)
            ldj = ldj + tf.reduce_sum(s, axis=(1, 2, 3), keepdims=True)
        return z, ldj
