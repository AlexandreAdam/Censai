import tensorflow as tf
from censai.models import UnetModelv2
from censai.definitions import DTYPE, logit, lrelu4p
from censai import PhysicalModelv2
from censai.utils import nulltape


class RIMSourceUnetv2:
    def __init__(
            self,
            physical_model: PhysicalModelv2,
            unet: UnetModelv2,
            steps: int,
            adam=True,
            source_link="identity",
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8,
            source_init=1e-3,
            flux_lagrange_multiplier: float=1.
    ):
        self.physical_model = physical_model
        self.source_pixels = physical_model.src_pixels
        self.unet = unet
        self.steps = steps
        self.adam = adam
        self._source_link_func = source_link
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._source_init = source_init
        self.flux_lagrange_multiplier = flux_lagrange_multiplier

        if self._source_link_func == "exp":
            self.source_inverse_link = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-6))
            self.source_link = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))
        elif self._source_link_func == "identity":
            self.source_inverse_link = tf.identity
            self.source_link = tf.identity
        elif self._source_link_func == "relu":
            self.source_inverse_link = tf.identity
            self.source_link = tf.nn.relu
        elif self._source_link_func == "sigmoid":
            self.source_inverse_link = logit
            self.source_link = tf.nn.sigmoid
        elif self._source_link_func == "leaky_relu":
            self.source_inverse_link = tf.identity
            self.source_link = tf.nn.leaky_relu
        elif self._source_link_func == "lrelu4p":
            self.source_inverse_link = tf.identity
            self.source_link = lrelu4p
        else:
            raise NotImplementedError(
                f"{source_link} not in ['exp', 'identity', 'relu', 'leaky_relu', 'lrelu4p', 'sigmoid']")

        if adam:
            self.grad_update = self.adam_grad_update
        else:
            self.grad_update = lambda x, t: x

    def adam_grad_update(self, grad, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_mean = self.beta_1 * self._grad_mean + (1 - self.beta_1) * grad
        self._grad_var = self.beta_2 * self._grad_var + (1 - self.beta_2) * tf.square(grad)
        # for grad update, unbias the moments
        m_hat = self._grad_mean / (1 - self.beta_1 ** (time_step + 1))
        v_hat = self._grad_var / (1 - self.beta_2 ** (time_step + 1))
        return m_hat / (tf.sqrt(v_hat) + self.epsilon)

    def initial_states(self, batch_size):
        # Define initial guess in physical space, then apply inverse link function to bring them in prediction space
        source_init = tf.ones(shape=(batch_size, self.source_pixels, self.source_pixels, 1)) * self._source_init
        states = self.unet.init_hidden_states(self.source_pixels, batch_size)

        # reset adam gradients
        self._grad_mean = tf.zeros_like(source_init, dtype=DTYPE)
        self._grad_var = tf.zeros_like(source_init, dtype=DTYPE)
        return source_init,  states

    def time_step(self, source, grad, states):
        source, states = self.unet(xt=source, grad=grad, states=states)
        return source, states

    def __call__(self, lensed_image, noise_rms, psf, outer_tape=nulltape):
        return self.call(lensed_image, noise_rms, psf, outer_tape)

    def call(self, lensed_image, kappa, noise_rms, psf, outer_tape=nulltape):
        """
        Used in training.
        """
        batch_size = lensed_image.shape[0]
        source, states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    y_pred = self.physical_model.forward(self.source_link(source), kappa, psf)
                    flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
                    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None]**2, axis=(1, 2, 3))
                    cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
                grad = g.gradient(cost, source)
                grad = self.grad_update(grad, current_step)
            source, states = self.time_step(source, grad, states)
            source_series = source_series.write(index=current_step, value=source)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step-1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa, psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return source_series.stack(),  chi_squared_series.stack()

    @tf.function
    def call_function(self, lensed_image, kappa, noise_rms, psf):
        """
        Used in training. Return linked source maps.

        This method use the tensorflow function autograph decorator, which enables us to use tf.gradients instead
        of creating a tape at each time steps. Potentially faster, but also memory hungry because for loop is unrolled
        when the graph is created.
        """
        batch_size = lensed_image.shape[0]
        source, states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            y_pred = self.physical_model.forward(self.source_link(source), kappa, psf)
            flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
            log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
            grad = tf.gradients(cost, source)
            grad = self.grad_update(grad, current_step)
            source, states = self.time_step(source, grad, states)
            source_series = source_series.write(index=current_step, value=source)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step-1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa, psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return source_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, kappa, noise_rms, psf):
        """
        Used in inference. Return physical kappa and source maps.
        """
        batch_size = lensed_image.shape[0]
        source, states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with tf.GradientTape() as g:
                g.watch(source)
                y_pred = self.physical_model.forward(self.source_link(source), kappa, psf)
                flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
                log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
                cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
            grad = g.gradient(cost, source)
            grad = self.grad_update(grad, current_step)
            source, states = self.time_step(source, grad, states)
            source_series = source_series.write(index=current_step, value=self.source_link(source))
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa, psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return source_series.stack(), chi_squared_series.stack()  # stack along 0-th dimension

    def cost_function(self, lensed_image, source, kappa, noise_rms, psf, outer_tape=nulltape, reduction=True):
        """

        Args:
            lensed_image: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        source_series, chi_squared = self.call(lensed_image, kappa, psf=psf, noise_rms=noise_rms, outer_tape=outer_tape)
        source_cost = tf.reduce_sum(tf.square(self.source_link(source_series) - source), axis=0) / self.steps
        chi = tf.reduce_sum(chi_squared, axis=0) / self.steps

        if reduction:
            return tf.reduce_mean(source_cost), tf.reduce_mean(chi)
        else:
            return tf.reduce_mean(source_cost, axis=(1, 2, 3)), chi
