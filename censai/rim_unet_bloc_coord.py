import tensorflow as tf
from censai.models import UnetModelv2
from censai.definitions import logkappa_normalization, log_10, DTYPE, lrelu4p
from censai import PhysicalModel
from censai.utils import nulltape


class RIMUnetBlocCoord:
    def __init__(
            self,
            physical_model: PhysicalModel,
            source_model: UnetModelv2,
            kappa_model: UnetModelv2,
            steps: int,
            adam=True,
            kappalog=True,
            kappa_normalize=True,
            source_link='identity',
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8,
            source_init=1.,
            kappa_init=1e-1
    ):
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.kappa_pixels
        self.source_pixels = physical_model.src_pixels
        self.steps = steps
        self.source_model = source_model
        self.kappa_model = kappa_model
        self.adam = adam
        self.kappalog = kappalog
        self.kappa_normalize = kappa_normalize
        self._source_link_func = source_link
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._kappa_init = kappa_init
        self._source_init = source_init

        if self.kappalog:
            if self.kappa_normalize:
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: logkappa_normalization(log_10(x), forward=True))
                self.kappa_link = tf.keras.layers.Lambda(lambda x: 10**(logkappa_normalization(x, forward=False)))
            else:
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: log_10(x))
                self.kappa_link = tf.keras.layers.Lambda(lambda x: 10**x)

        else:
            self.kappa_link = tf.identity
            self.kappa_inverse_link = tf.identity

        if self._source_link_func == "exp":
            self.source_link = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))
        elif self._source_link_func == "identity":
            self.source_link = tf.identity
        elif self._source_link_func == "relu":
            self.source_link = tf.nn.relu
        elif self._source_link_func == "sigmoid":
            self.source_link = tf.nn.sigmoid
        elif self._source_link_func == "leaky_relu":
            self.source_link = tf.nn.leaky_relu
        elif self._source_link_func == "lrelu4p":
            self.source_link = lrelu4p
        else:
            raise NotImplementedError(f"{source_link} not in ['exp', 'identity', 'relu', 'leaky_relu', 'lrelu4p', 'sigmoid']")

        if adam:
            self.source_grad_update = self.adam_source_grad_update
            self.kappa_grad_update = self.adam_kappa_grad_update
        else:
            self.source_grad_update = tf.keras.layers.Lambda(lambda x, t: x)
            self.kappa_grad_update = tf.keras.layers.Lambda(lambda x, t: x)

    def initial_states(self, batch_size):
        # Define initial guess in physical space, then apply inverse link function to bring them in prediction space
        source_init = tf.ones(shape=(batch_size, self.source_pixels, self.source_pixels, 1)) * self._source_init
        kappa_init = self.kappa_inverse_link(tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1)) * self._kappa_init)

        source_states = self.source_model.init_hidden_states(self.source_pixels, batch_size)
        kappa_states = self.kappa_model.init_hidden_states(self.kappa_pixels, batch_size)

        # reset adam gradients
        self._grad_mean1 = tf.zeros_like(source_init, dtype=DTYPE)
        self._grad_var1 = tf.zeros_like(source_init, dtype=DTYPE)
        self._grad_mean2 = tf.zeros_like(kappa_init, dtype=DTYPE)
        self._grad_var2 = tf.zeros_like(kappa_init, dtype=DTYPE)
        return source_init, source_states, kappa_init, kappa_states

    def adam_source_grad_update(self, grad, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_mean1 = self.beta_1 * self._grad_mean1 + (1 - self.beta_1) * grad
        self._grad_var1 = self.beta_2 * self._grad_var1 + (1 - self.beta_2) * tf.square(grad)
        # for grad update, unbias the moments
        m_hat1 = self._grad_mean1 / (1 - self.beta_1 ** (time_step + 1))
        v_hat1 = self._grad_var1 / (1 - self.beta_2 ** (time_step + 1))
        return m_hat1 / (tf.sqrt(v_hat1) + self.epsilon)

    def adam_kappa_grad_update(self, grad, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_mean2 = self.beta_1 * self._grad_mean2 + (1 - self.beta_1) * grad
        self._grad_var2 = self.beta_2 * self._grad_var2 + (1 - self.beta_2) * tf.square(grad)
        # for grad update, unbias the moments
        m_hat2 = self._grad_mean2 / (1 - self.beta_1 ** (time_step + 1))
        v_hat2 = self._grad_var2 / (1 - self.beta_2 ** (time_step + 1))
        return m_hat2 / (tf.sqrt(v_hat2) + self.epsilon)

    def kappa_time_step(self, y_true, source, kappa, kappa_states, current_step, outer_tape=nulltape):
        with outer_tape.stop_recording():
            with tf.GradientTape() as g:
                g.watch(kappa)
                log_likelihood = self.physical_model.log_likelihood(y_true=y_true, source=self.source_link(source), kappa=self.kappa_link(kappa))
                cost = tf.reduce_mean(log_likelihood)
            kappa_grad = g.gradient(cost, kappa)
            kappa_grad = self.kappa_grad_update(kappa_grad, current_step)
        kappa, kappa_states = self.kappa_model(kappa, kappa_states, kappa_grad)
        return kappa, kappa_states, log_likelihood

    def source_time_step(self, y_true, source, kappa, source_states, current_step, outer_tape=nulltape):
        with outer_tape.stop_recording():
            with tf.GradientTape() as g:
                g.watch(source)
                log_likelihood = self.physical_model.log_likelihood(y_true=y_true, source=self.source_link(source), kappa=self.kappa_link(kappa))
                cost = tf.reduce_mean(log_likelihood)
            source_grad = g.gradient(cost, source)
            source_grad = self.source_grad_update(source_grad, current_step)
        source, source_states = self.source_model(source, source_states, source_grad)
        return source, source_states, log_likelihood

    def __call__(self, lensed_image, outer_tape=nulltape):
        return self.call(lensed_image, outer_tape)

    def call(self, lensed_image, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, source_states, kappa, kappa_states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            kappa, kappa_states, _ = self.kappa_time_step(lensed_image, source, kappa, kappa_states, current_step, outer_tape)
            source, source_states, log_likelihood = self.source_time_step(lensed_image, source, kappa, source_states, current_step, outer_tape)
            source_series = source_series.write(index=current_step, value=source)
            kappa_series = kappa_series.write(index=current_step, value=kappa)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=self.kappa_link(kappa))
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, source_states, kappa, kappa_states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            kappa, kappa_states, _ = self.kappa_time_step(lensed_image, source, kappa, kappa_states, current_step, outer_tape)
            source, source_states, log_likelihood = self.source_time_step(lensed_image, source, kappa, source_states, current_step, outer_tape)
            source_series = source_series.write(index=current_step, value=self.source_link(source))
            kappa_series = kappa_series.write(index=current_step, value=self.kappa_link(kappa))
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=self.kappa_link(kappa))
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()
