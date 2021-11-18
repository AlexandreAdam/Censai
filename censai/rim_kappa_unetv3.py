import tensorflow as tf
from censai.models import UnetModelv2
from censai.definitions import logkappa_normalization, DTYPE
from censai import PhysicalModelv2
from censai.utils import nulltape

log_10 = lambda x: tf.math.log(x) / tf.math.log(10.)


class RIMKappaUnetv2:
    def __init__(
            self,
            physical_model: PhysicalModelv2,
            unet: UnetModelv2,
            kappa_init: tf.Tensor,
            steps: int,
            adam=True,
            kappalog=True,
            kappa_normalize=False,
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8,
    ):
        assert len(kappa_init.shape) == 4
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.kappa_pixels
        self.unet = unet
        self.steps = steps
        self.adam = adam
        self.kappalog = kappalog
        self.kappa_normalize = kappa_normalize
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.kappa_init = kappa_init

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
        kappa_init = tf.tile(self.kappa_inverse_link(self.kappa_init), [batch_size, 1, 1, 1])
        states = self.unet.init_hidden_states(self.kappa_pixels, batch_size)

        # reset adam gradients
        self._grad_mean = tf.zeros_like(kappa_init, dtype=DTYPE)
        self._grad_var = tf.zeros_like(kappa_init, dtype=DTYPE)
        return kappa_init, states

    def time_step(self, kappa, grad, states):
        kappa, states = self.unet(xt=kappa, grad=grad, states=states)
        return kappa, states

    def __call__(self, lensed_image, noise_rms, psf, outer_tape=nulltape):
        return self.call(lensed_image, noise_rms, psf, outer_tape)

    def call(self, lensed_image, source, noise_rms, psf, outer_tape=nulltape):
        """
        Used in training. Return linked kappa and source maps.
        """
        batch_size = lensed_image.shape[0]
        kappa, states = self.initial_states(batch_size)

        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(kappa)
                    y_pred = self.physical_model.forward(source, self.kappa_link(kappa), psf)
                    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None]**2, axis=(1, 2, 3))
                    cost = tf.reduce_mean(log_likelihood)
                grad = g.gradient(cost, kappa)
                grad = self.grad_update(grad, current_step)
            kappa, states = self.time_step(kappa, grad, states)
            kappa_series = kappa_series.write(index=current_step, value=kappa)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step-1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return kappa_series.stack(), chi_squared_series.stack()

    @tf.function
    def call_function(self, lensed_image, source, noise_rms, psf):
        """
        Used in training.

        This method use the tensorflow function autograph decorator, which enables us to use tf.gradients instead
        of creating a tape at each time steps. Potentially faster, but also memory hungry because for loop is unrolled
        when the graph is created.
        """
        batch_size = lensed_image.shape[0]
        kappa, states= self.initial_states(batch_size)

        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in tf.range(self.steps):
            y_pred = self.physical_model.forward(source, self.kappa_link(kappa), psf)
            log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = tf.reduce_mean(log_likelihood)
            grad = tf.gradients(cost, kappa)
            grad = self.grad_update(grad, current_step)
            kappa, states = self.time_step(kappa, grad, states)
            kappa_series = kappa_series.write(index=current_step, value=kappa)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step-1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return kappa_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, source, noise_rms, psf):
        """
        Used in inference. Return physical kappa maps.
        """
        batch_size = lensed_image.shape[0]
        kappa, states = self.initial_states(batch_size)

        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with tf.GradientTape() as g:
                g.watch(kappa)
                y_pred = self.physical_model.forward(source, self.kappa_link(kappa), psf)
                log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
                cost = tf.reduce_mean(log_likelihood)
            grad = g.gradient(cost, kappa)
            grad = self.grad_update(grad, current_step)
            kappa, states = self.time_step(kappa, grad, states)
            kappa_series = kappa_series.write(index=current_step, value=self.kappa_link(kappa))
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return kappa_series.stack(), chi_squared_series.stack()  # stack along 0-th dimension

    def cost_function(self, lensed_image, source, kappa, noise_rms, psf, outer_tape=nulltape, reduction=True):
        """

        Args:
            lensed_image: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        kappa_series, chi_squared = self.call(lensed_image, source, psf=psf, noise_rms=noise_rms, outer_tape=outer_tape)
        kappa_cost = tf.reduce_sum(tf.square(kappa_series - self.kappa_inverse_link(kappa)), axis=0) / self.steps
        chi = tf.reduce_sum(chi_squared, axis=0) / self.steps

        if reduction:
            return tf.reduce_mean(kappa_cost), tf.reduce_mean(chi)
        else:
            return tf.reduce_mean(kappa_cost, axis=(1, 2, 3)), chi

