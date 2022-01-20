import tensorflow as tf
from censai.models import SharedUnetModelv4
from censai.definitions import logkappa_normalization, log_10, DTYPE, logit, lrelu4p
from censai import PhysicalModelv2
from censai.utils import nulltape


class RIMSharedUnetv3:
    def __init__(
            self,
            physical_model: PhysicalModelv2,
            unet: SharedUnetModelv4,
            steps: int,
            adam=True,
            rmsprop=True, # overwrite ADAM for now
            kappalog=True,
            kappa_normalize=False,
            source_link="relu",
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8,
            flux_lagrange_multiplier: float=0.
    ):
        self.physical_model = physical_model
        self.pixels = physical_model.kappa_pixels
        self.unet = unet
        self.steps = steps
        self.adam = adam
        self.kappalog = kappalog
        self._source_link_func = source_link
        self.kappa_normalize = kappa_normalize
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.flux_lagrange_multiplier = flux_lagrange_multiplier

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

        if rmsprop:
            self.grad_update = self.rmsprop_grad_update
        elif adam:
            self.grad_update = self.adam_grad_update
        else:
            self.grad_update = lambda x, y, t: (x, y)

    def adam_grad_update(self, grad1, grad2, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_mean1 = self.beta_1 * self._grad_mean1 + (1 - self.beta_1) * grad1
        self._grad_var1 = self.beta_2 * self._grad_var1 + (1 - self.beta_2) * tf.square(grad1)
        self._grad_mean2 = self.beta_1 * self._grad_mean2 + (1 - self.beta_1) * grad2
        self._grad_var2 = self.beta_2 * self._grad_var2 + (1 - self.beta_2) * tf.square(grad2)
        # Unbias the moments
        m_hat1 = self._grad_mean1 / (1 - self.beta_1 ** (time_step + 1))
        v_hat1 = self._grad_var1 / (1 - self.beta_2 ** (time_step + 1))
        m_hat2 = self._grad_mean2 / (1 - self.beta_1 ** (time_step + 1))
        v_hat2 = self._grad_var2 / (1 - self.beta_2 ** (time_step + 1))
        return m_hat1 / (tf.sqrt(v_hat1) + self.epsilon), m_hat2 / (tf.sqrt(v_hat2) + self.epsilon)

    def rmsprop_grad_update(self, grad1, grad2, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_var1 = self.beta_1 * self._grad_var1 + (1 - self.beta_1) * tf.square(grad1)
        self._grad_var2 = self.beta_1 * self._grad_var2 + (1 - self.beta_1) * tf.square(grad2)
        # Unbias the moments
        v_hat1 = self._grad_var1 / (1 - self.beta_1 ** (time_step + 1))
        v_hat2 = self._grad_var2 / (1 - self.beta_1 ** (time_step + 1))
        return grad1 / (tf.sqrt(v_hat1) + self.epsilon), grad2 / (tf.sqrt(v_hat2) + self.epsilon)

    def initial_states(self, batch_size):
        # Define initial guess in physical space, then apply inverse link function to bring them in prediction space
        source_init = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1], dtype=DTYPE)
        kappa_init = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1], dtype=DTYPE)
        source_grad = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1], dtype=DTYPE)
        kappa_grad = tf.zeros(shape=[batch_size, self.pixels, self.pixels, 1], dtype=DTYPE)

        states = self.unet.init_hidden_states(self.pixels, batch_size)

        # reset adam gradients
        self._grad_mean1 = tf.zeros_like(source_init, dtype=DTYPE)
        self._grad_var1 = tf.zeros_like(source_init, dtype=DTYPE)
        self._grad_mean2 = tf.zeros_like(kappa_init, dtype=DTYPE)
        self._grad_var2 = tf.zeros_like(kappa_init, dtype=DTYPE)
        return source_init, kappa_init, source_grad, kappa_grad, states

    def time_step(self, lens, source, kappa, source_grad, kappa_grad, states, training=True):
        x = tf.concat([lens, source, kappa, source_grad, kappa_grad], axis=3)
        delta_xt, states = self.unet(x, states, training=training)
        delta_source, delta_kappa = tf.split(delta_xt, 2, axis=-1)
        source = source + delta_source
        kappa = kappa + delta_kappa
        return source, kappa, states

    def __call__(self, lensed_image, noise_rms, psf, outer_tape=nulltape):
        return self.call(lensed_image, noise_rms, psf, outer_tape)

    def call(self, lensed_image, noise_rms, psf, outer_tape=nulltape):
        """
        Used in training. Return linked kappa and source maps.
        """
        batch_size = lensed_image.shape[0]
        source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
        source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)  # Use lens to make an initial guess with Unet
        source_series = tf.TensorArray(DTYPE, size=self.steps)
        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        # record initial guess
        source_series = source_series.write(index=0, value=source)
        kappa_series = kappa_series.write(index=0, value=kappa)
        # Main optimization loop
        for current_step in tf.range(self.steps-1):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    g.watch(kappa)
                    y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
                    flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
                    log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None]**2, axis=(1, 2, 3))
                    cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
                source_grad, kappa_grad = g.gradient(cost, [source, kappa])
                source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
            source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)
            source_series = source_series.write(index=current_step+1, value=source)
            kappa_series = kappa_series.write(index=current_step+1, value=kappa)
            chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood/self.pixels**2)  # renormalize chi squared here
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()

    @tf.function
    def call_function(self, lensed_image, noise_rms, psf):
        """
        Used in training. Return linked kappa and source maps.

        This method use the tensorflow function autograph decorator, which enables us to use tf.gradients instead
        of creating a tape at each time steps. Potentially faster, but also memory hungry because for loop is unrolled
        when the graph is created.
        """
        batch_size = lensed_image.shape[0]
        source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
        source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)  # Use lens to make an initial guess with Unet
        source_series = tf.TensorArray(DTYPE, size=self.steps)
        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        # record initial guess
        source_series = source_series.write(index=0, value=source)
        kappa_series = kappa_series.write(index=0, value=kappa)
        # Main optimization loop
        for current_step in tf.range(self.steps-1):
            y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
            flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
            log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
            cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
            source_grad, kappa_grad = tf.gradients(cost, [source, kappa])
            source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
            source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)
            source_series = source_series.write(index=current_step+1, value=source)
            kappa_series = kappa_series.write(index=current_step+1, value=kappa)
            chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood/self.pixels**2)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood) # chi squared is normalized in physical model
        return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, noise_rms, psf):
        """
        Used in inference. Return physical kappa and source maps.
        """
        batch_size = lensed_image.shape[0]
        source, kappa, source_grad, kappa_grad, states = self.initial_states(batch_size)  # initiate all tensors to 0
        source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states)  # Use lens to make an initial guess with Unet
        source_series = tf.TensorArray(DTYPE, size=self.steps)
        kappa_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        # record initial guess
        source_series = source_series.write(index=0, value=self.source_link(source))
        kappa_series = kappa_series.write(index=0, value=self.kappa_link(kappa))
        # Main optimization loop
        for current_step in range(self.steps-1):
            with tf.GradientTape() as g:
                g.watch(source)
                g.watch(kappa)
                y_pred = self.physical_model.forward(self.source_link(source), self.kappa_link(kappa), psf)
                flux_term = tf.square(tf.reduce_sum(y_pred, axis=(1, 2, 3)) - tf.reduce_sum(lensed_image, axis=(1, 2, 3)))
                log_likelihood = 0.5 * tf.reduce_sum(tf.square(y_pred - lensed_image) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
                cost = log_likelihood + self.flux_lagrange_multiplier * flux_term
            source_grad, kappa_grad = g.gradient(cost, [source, kappa])
            source_grad, kappa_grad = self.grad_update(source_grad, kappa_grad, current_step)
            source, kappa, states = self.time_step(lensed_image, source, kappa, source_grad, kappa_grad, states, training=False)
            source_series = source_series.write(index=current_step+1, value=self.source_link(source))
            kappa_series = kappa_series.write(index=current_step+1, value=self.kappa_link(kappa))
            chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood/self.pixels**2)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=self.kappa_link(kappa), psf=psf, noise_rms=noise_rms)
        chi_squared_series = chi_squared_series.write(index=self.steps-1, value=log_likelihood)
        return source_series.stack(), kappa_series.stack(), chi_squared_series.stack()  # stack along 0-th dimension

    def cost_function(self, lensed_image, source, kappa, noise_rms, psf, outer_tape=nulltape, reduction=True):
        """

        Args:
            lensed_image: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        source_series, kappa_series, chi_squared = self.call(lensed_image, psf=psf, noise_rms=noise_rms, outer_tape=outer_tape)
        source_cost = tf.reduce_sum(tf.square(source_series - self.source_inverse_link(source)), axis=0) / self.steps
        kappa_cost = tf.reduce_sum(tf.square(kappa_series - self.kappa_inverse_link(kappa)), axis=0) / self.steps
        chi = tf.reduce_sum(chi_squared, axis=0) / self.steps

        if reduction:
            return tf.reduce_mean(source_cost) + tf.reduce_mean(kappa_cost), tf.reduce_mean(chi)
        else:
            return tf.reduce_mean(source_cost, axis=(1, 2, 3)) + tf.reduce_mean(kappa_cost, axis=(1, 2, 3)), chi

