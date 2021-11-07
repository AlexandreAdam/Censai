import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class SqueezeFlow(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.prior = tfd.Normal(loc=0.0, scale=1.0)

    def __call__(self, z, ldj, reverse=False):
        if reverse:
            z_split = self.prior.sample(sample_shape=z.shape)
            z = tf.concat([z, z_split], axis=-1)
            ldj -= tf.reduce_sum(self.prior.log_prob(z_split), axis=(1, 2, 3))
        else:
            z, z_split = tf.split(z, 2, axis=-1)
            ldj += tf.reduce_sum(self.prior.log_prob(z_split), axis=(1, 2, 3))
        return z, ldj

