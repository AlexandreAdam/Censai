import tensorflow as tf
from censai.models import UnetModel
from censai.definitions import logkappa_normalization, log_10, kappa_clipped_exponential
from censai import PhysicalModel
from censai.utils import nulltape


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
            source_link='identity',
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            source_init=1e-3,
            kappa_init=1e-1
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
        self._source_link_func = source_link
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._kappa_init = kappa_init
        self._source_init = source_init

        if self.kappalog:
            if self.kappa_normalize:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: logkappa_normalization(log_10(x), forward=True))
                # self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: kappa_clipped_exponential(logkappa_normalization(x, forward=False)))
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**(logkappa_normalization(x, forward=False)))
            else:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: log_10(x))
                # self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: kappa_clipped_exponential(x))
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**x)

        else:
            self.kappa_link = tf.identity
            self.kappa_inverse_link = tf.identity

        if self._source_link_func == "exp":
            self.source_link = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-6))
            self.source_inverse_link = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))
        elif self._source_link_func == "sqrt":
            self.source_link = tf.keras.layers.Lambda(lambda x: tf.math.sqrt(x + 1e-6))
            self.source_inverse_link = tf.keras.layers.Lambda(lambda x: x**2)
        elif self._source_link_func == "identity":
            self.source_link = tf.identity
            self.source_inverse_link = tf.identity
        else:
            raise NotImplementedError(f"{source_link} not in ['exp', 'sqrt', 'identity']")

    def initial_states(self, batch_size):
        source_init = self.source_link(tf.ones(shape=(batch_size, self.source_pixels, self.source_pixels, 1)) * self._source_init)
        kappa_init = self.kappa_link(tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1)) * self._kappa_init)

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

    def time_step(self, sources, source_states, source_grad, kappa, kappa_states, kappa_grad, scope=None):
        new_source, new_source_states = self.source_model(sources, source_states, source_grad)
        new_kappa, new_kappa_states = self.kappa_model(kappa, kappa_states, kappa_grad)
        return new_source, new_source_states, new_kappa, new_kappa_states

    def __call__(self, lensed_image, outer_tape=nulltape):
        return self.call(lensed_image, outer_tape)

    def call(self, lensed_image, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, source_states, kappa, kappa_states = self.initial_states(batch_size)

        source_series = []
        kappa_series = []
        chi_squared_series = []
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    g.watch(kappa)
                    cost = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_inverse_link(source), kappa=self.kappa_inverse_link(kappa))
            source_grad, kappa_grad = g.gradient(cost, [source, kappa])
            source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
            source, source_states, kappa, kappa_states = self.time_step(source, source_states, source_grad, kappa, kappa_states, kappa_grad)
            source_series.append(source)
            kappa_series.append(kappa)
            chi_squared_series.append(cost)
        return source_series, kappa_series, chi_squared_series

    def predict(self, lensed_image):
        batch_size = lensed_image.shape[0]
        source, source_states, kappa, kappa_states = self.initial_states(batch_size)

        source_series = []
        kappa_series = []
        chi_squared_series = []
        for current_step in range(self.steps):
            with tf.GradientTape() as g:
                g.watch(source)
                g.watch(kappa)
                cost = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_inverse_link(source), kappa=self.kappa_inverse_link(kappa))
            source_grad, kappa_grad = g.gradient(cost, [source, kappa])
            source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
            source, source_states, kappa, kappa_states = self.time_step(source, source_states, source_grad, kappa, kappa_states, kappa_grad)
            source_series.append(self.source_inverse_link(source))
            kappa_series.append(self.kappa_inverse_link(kappa))
            chi_squared_series.append(cost)
        return source_series, kappa_series, chi_squared_series

    def cost_function(self, lensed_image, source, kappa, outer_tape=nulltape, reduction=True):
        """

        Args:
            lensed_image: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        source_series, kappa_series, _ = self.call(lensed_image, outer_tape=outer_tape)
        # chi1 = sum([tf.square(source_series[i] - self.source_link(source)) for i in range(self.steps)]) / self.steps
        chi1 = sum([tf.square(source_series[i] - self.source_link(source)) for i in range(self.steps)]) / self.steps
        chi2 = sum([tf.square(kappa_series[i] - self.kappa_link(kappa)) for i in range(self.steps)]) / self.steps
        if reduction:
            return tf.reduce_mean(chi1) + tf.reduce_mean(chi2)
        else:
            return tf.reduce_mean(chi1, axis=(1, 2, 3)) + tf.reduce_mean(chi2, axis=(1, 2, 3))
