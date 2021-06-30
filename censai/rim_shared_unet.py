import tensorflow as tf
from censai.models import UnetModel
from censai.definitions import logkappa_normalization, log_kappa
from censai import PhysicalModel


def is_power_of_two(n):
    while n > 1:
        n /= 2
        if int(n) != n:
            return False
        elif n == 2:
            return True

class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(self, use_bilinear: bool, layers: int, strides: int):
        super(UpsamplingLayer, self).__init__()
        self.upsamplings = []
        for i in range(layers):
            if use_bilinear:
                self.upsamplings.append(
                    tf.keras.layers.UpSampling2D(strides)
                )

    def call(self, kappa):


class RIMSharedUnet:
    """
    Architecture has only 1 Unet. Source and kappa information are stacked along channel dimension.

    There are 2 intended structures:
        1. Kappa has a larger shape than Source tensor:
            1 - Use a half-strided convolution to upsample the output of the Unet
            3 - Use bilinear interpolation to upsample
        2. Kappa and Source have the same tensor shape -> Identity layer

    In any case, we use the Source shape for the Unet
    """
    def __init__(
            self,
            physical_model: PhysicalModel,
            unet: UnetModel,
            steps: int,
            adam=True,
            kappalog=True,
            kappa_normalize=True,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
    ):
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.pixels
        self.source_pixels = physical_model.src_pixels
        self.unet = unet
        self.steps = steps
        self.adam = adam
        self.kappalog = kappalog
        self.kappa_normalize = kappa_normalize
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        assert is_power_of_two(self.kappa_pixels)
        assert is_power_of_two(self.source_pixels)

        if self.source_pixels
        self.kappa_resample_layers


    def initial_states(self, batch_size):
        source_init = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, 1))
        if self.kappalog:
            if self.kappa_normalize:
                kappa_init = tf.zeros(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
            else:
                kappa_init = -tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
        else:
            kappa_init = tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1)) / 10

        init_tensor = tf.concat([source_init, kappa_init], axis=-1)
        states = self.unet.init_hidden_states(self.source_pixels, batch_size)
        return init_tensor, states

    @tf.function
    def kappa_link(self, kappa):
        if self.kappalog:
            kappa = log_kappa(kappa)
            if self.kappa_normalize:
                return logkappa_normalization(kappa, forward=True)
            return kappa
        else:
            return kappa

    @tf.function
    def kappa_inverse_link(self, eta):
        if self.kappalog:
            if self.kappa_normalize:
                eta = logkappa_normalization(eta, forward=False)
            return 10**(eta)
        else:
            return eta

    def resize_kappa(self, kappa):


    def grad_update(self, grad, time_step):
        if self.adam:
            if time_step == 0:  # reset mean and variance for time t=-1
                self._grad_mean1 = tf.zeros_like(grad)
                self._grad_var1 = tf.zeros_like(grad)
            self._grad_mean1 = self. beta_1 * self._grad_mean1 + (1 - self.beta_1) * grad
            self._grad_var1  = self.beta_2 * self._grad_var1 + (1 - self.beta_2) * tf.square(grad)
            # for grad update, unbias the moments
            m_hat1 = self._grad_mean1 / (1 - self.beta_1**(time_step + 1))
            v_hat1 = self._grad_var1 / (1 - self.beta_2**(time_step + 1))
            return m_hat1 / (tf.sqrt(v_hat1) + self.epsilon)
        else:
            return grad

    def time_step(self, xt, states, grad, scope=None):
        xt_1, ht_1 = self.unet(xt, states, grad)
        return xt_1, ht_1

    def __call__(self, lensed_image):
        return self.call(lensed_image)

    def call(self, lensed_image):
        batch_size = lensed_image.shape[0]
        X, states = self.initial_states(batch_size)

        source_series = []
        kappa_series  = []
        with tf.GradientTape() as g:
            g.watch(X)
            source, kappa = tf.split(X, 2, axis=-1)
            cost = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=self.kappa_inverse_link(kappa))
        grad = g.gradient(cost, X)
        grad = self.grad_update(grad, 0)

        X, states = self.time_step(X, states, grad)
        source, kappa = tf.split(X, 2, axis=-1)
        source_series.append(source)
        kappa_series.append(kappa)

        for current_step in range(1, self.steps):
            with tf.GradientTape() as g:
                g.watch(X)
                source, kappa = tf.split(X, 2, axis=-1)
                cost = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=self.kappa_inverse_link(kappa))
            grad = g.gradient(cost, X)
            grad = self.grad_update(grad, current_step)
            X, states = self.time_step(X, states, grad)
            source, kappa = tf.split(X, 2, axis=-1)
            source_series.append(source)
            kappa_series.append(kappa)
        return source_series, kappa_series, cost

    def cost_function(self, lensed_image, source, kappa, reduction=True):
        """

        Args:
            lensed_image: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        source_series, kappa_series, _ = self.call(lensed_image)
        chi1 = sum([tf.square(source_series[i] - source) for i in range(self.steps)]) / self.steps
        chi2 = sum([tf.square(kappa_series[i] - self.kappa_link(kappa)) for i in range(self.steps)]) / self.steps
        if reduction:
            return tf.reduce_mean(chi1) + tf.reduce_mean(chi2)
        else:
            return tf.reduce_mean(chi1, axis=(1, 2, 3)) + tf.reduce_mean(chi2, axis=(1, 2, 3))

