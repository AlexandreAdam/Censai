import tensorflow as tf
import tensorflow_probability as tfp
from censai.definitions import DTYPE

tfd = tfp.distributions


class ImageFlow(tf.keras.Model):
    def __init__(self, flows, pixels, importance_samples=8):
        super(ImageFlow, self).__init__()
        self.flows = flows
        self.importance_samples = importance_samples
        self.prior = tfd.MultivariateNormalDiag(tf.zeros(shape=(pixels, pixels, 1)), tf.ones(shape=(pixels, pixels, 1)))
        self.pixels = pixels

    def forward(self, x):
        return self._get_likelihood(x)

    def encode(self, x):
        batch_size = x.shape[0]
        z, ldj = x, tf.zeros(shape=batch_size)
        for flow in self.flows:
            z, ldj = flow(z, ldj, reverse=False)
        return z, ldj

    def _get_likelihood(self, x, return_ll=False):
        z, ldj = self.encode(x)  # return latent space vector z and
        log_pz = tf.reduce_sum(self.prior.log_prob(z), axis=(1, 2, 3))
        log_px = ldj + log_pz
        return log_px if return_ll else tf.reduce_mean(-log_px)

    def sample(self, batch_size, z_init=None, seed=None):
        if z_init is None:
            z = self.prior.sample(batch_size, seed)
        else:
            z = z_init
        ldj = tf.zeros(shape=batch_size)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def training_step(self, x):
        loss = self._get_likelihood(x)
        return loss

    def test_step(self, x):
        samples = []
        for _ in range(self.importance_samples):
            ll = self._get_likelihood(x, return_ll=True)
            samples.append(ll)
        ll = tf.stack(samples, axis=-1)

        ll = tf.math.reduce_logsumexp(ll, axis=-1) - tf.math.log(tf.cast(self.importance_samples, DTYPE))
        return tf.reduce_mean(-ll)
