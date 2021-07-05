import tensorflow as tf
from censai.models import UnetModel
from censai.definitions import logkappa_normalization, log_10, DTYPE, LOGFLOOR
from censai import PhysicalModel
# from scipy.signal import tukey
# import numpy as np


class RIMUnet:
    def __init__(
            self,
            physical_model: PhysicalModel,
            source_model: UnetModel,
            kappa_model: UnetModel,
            steps: int,
            adam=True,
            kappalog=True,
            kappa_normalize=True,
            source_tukey_alpha=0.6,
            # sourcelog=True,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
    ):
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.pixels
        self.source_pixels = physical_model.src_pixels
        self.steps = steps
        self.source_model = source_model
        self.kappa_model = kappa_model
        self.adam = adam
        self.kappalog = kappalog
        self.kappa_normalize = kappa_normalize
        self.tukey_alpha = source_tukey_alpha
        # self.sourcelog = sourcelog
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # window = tukey(128, alpha=0.6)
        # window = np.outer(window, window)
        # window = tf.constant(window[np.newaxis, ..., np.newaxis], dtype=DTYPE)

        if self.kappalog:
            if self.kappa_normalize:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: logkappa_normalization(log_10(x), forward=True))
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**logkappa_normalization(x, forward=False))
            else:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: log_10(x))
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**x)
        else:
            self.kappa_link = tf.identity
            self.kappa_inverse_link = tf.identity

        # if self.sourcelog:
        #     self.source_link = tf.keras.layers.Lambda(lambda x: tf.math.log(x + LOGFLOOR))
        #     self.source_inverse_link = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))
        # else:
        #     self.source_link = tf.identity
        #     self.source_inverse_link = tf.identity

    def initial_states(self, batch_size):
        # if self.sourcelog:
        #     source_init = tf.ones(shape=(batch_size, self.source_pixels, self.source_pixels, 1)) * tf.math.log(1e-3)
        # else:
        #     source_init = tf.ones(shape=(batch_size, self.source_pixels, self.source_pixels, 1)) / 1000

        source_init = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, 1))
        if self.kappalog:
            if self.kappa_normalize:
                kappa_init = tf.zeros(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
            else:
                kappa_init = -tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
        else:
            kappa_init = tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1)) / 10

        source_states = self.source_model.init_hidden_states(self.source_pixels, batch_size)
        kappa_states = self.kappa_model.init_hidden_states(self.kappa_pixels, batch_size)
        return source_init, source_states, kappa_init, kappa_states

    def grad_update(self, grad1, grad2, time_step):
        if self.adam:
            if time_step == 0:  # reset mean and variance for time t=-1
                self._grad_mean1 = tf.zeros_like(grad1)
                self._grad_var1 = tf.zeros_like(grad1)
                self._grad_mean2 = tf.zeros_like(grad2)
                self._grad_var2 = tf.zeros_like(grad2)
            self._grad_mean1 = self. beta_1 * self._grad_mean1 + (1 - self.beta_1) * grad1
            self._grad_var1  = self.beta_2 * self._grad_var1 + (1 - self.beta_2) * tf.square(grad1)
            self._grad_mean2 = self. beta_1 * self._grad_mean2 + (1 - self.beta_1) * grad2
            self._grad_var2  = self.beta_2 * self._grad_var2 + (1 - self.beta_2) * tf.square(grad2)
            # for grad update, unbias the moments
            m_hat1 = self._grad_mean1 / (1 - self.beta_1**(time_step + 1))
            v_hat1 = self._grad_var1 / (1 - self.beta_2**(time_step + 1))
            m_hat2 = self._grad_mean2 / (1 - self.beta_1**(time_step + 1))
            v_hat2 = self._grad_var2 / (1 - self.beta_2**(time_step + 1))
            return m_hat1 / (tf.sqrt(v_hat1) + self.epsilon), m_hat2 / (tf.sqrt(v_hat2) + self.epsilon)
        else:
            return grad1, grad2

    def time_step(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2, scope=None):
        xt_1, ht_1 = self.source_model(inputs_1, state_1, grad_1)
        xt_2, ht_2 = self.kappa_model(inputs_2, state_2, grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def __call__(self, lensed_image):
        return self.call(lensed_image)

    def call(self, lensed_image):
        batch_size = lensed_image.shape[0]
        source_init, state_1, kappa_init, state_2 = self.initial_states(batch_size)
        # 1=source, 2=kappa
        output_series_1 = []
        output_series_2 = []
        with tf.GradientTape() as g:
            g.watch(source_init)
            g.watch(kappa_init)
            cost = self.physical_model.log_likelihood(y_true=lensed_image,
                                                      # source=self.source_inverse_link(source_init),
                                                      source=source_init,
                                                      kappa=self.kappa_inverse_link(kappa_init))
        grads = g.gradient(cost, [source_init, kappa_init])
        grads = self.grad_update(*grads, 0)

        output_1, state_1, output_2, state_2 = self.time_step(source_init, state_1, grads[0], kappa_init, state_2, grads[1])
        output_series_1.append(output_1)
        output_series_2.append(output_2)

        for current_step in range(1, self.steps):
            with tf.GradientTape() as g:
                g.watch(output_1)
                g.watch(output_2)
                cost = self.physical_model.log_likelihood(y_true=lensed_image,
                                                          # source=self.source_inverse_link(output_1),
                                                          source=output_1,
                                                          kappa=self.kappa_inverse_link(output_2))
            grads = g.gradient(cost, [output_1, output_2])
            grads = self.grad_update(*grads, current_step)
            output_1, state_1, output_2, state_2 = self.time_step(output_1, state_1, grads[0], output_2, state_2, grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)
        return output_series_1, output_series_2, cost

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
        chi1 = sum([tf.square(source_series[i] - self.source_link(source)) for i in range(self.steps)]) / self.steps
        chi2 = sum([tf.square(kappa_series[i] - self.kappa_link(kappa)) for i in range(self.steps)]) / self.steps
        if reduction:
            return tf.reduce_mean(chi1) + tf.reduce_mean(chi2)
        else:
            return tf.reduce_mean(chi1, axis=(1, 2, 3)) + tf.reduce_mean(chi2, axis=(1, 2, 3))
