import tensorflow as tf
from censai.models import ModelMorningstar
from censai.definitions import DTYPE, lrelu4p
from censai import PhysicalModel
from censai.utils import nulltape


class RIM:
    def __init__(
            self,
            physical_model: PhysicalModel,
            model: ModelMorningstar,
            steps: int,
            adam=True,
            source_tukey_alpha=0.,
            source_link='sigmoid',
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8
    ):
        self.physical_model = physical_model
        self.source_pixels = physical_model.src_pixels
        self.steps = steps
        self.model = model
        self.adam = adam
        self.tukey_alpha = source_tukey_alpha
        self._source_link_func = source_link
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        if self._source_link_func == "exp":
            self.source_link = tf.math.exp
        elif self._source_link_func == "identity":
            self.source_link = tf.identity
        elif self._source_link_func == "relu":
            self.source_link = tf.nn.relu
        elif self._source_link_func == "sigmoid":
            self.source_link = tf.nn.sigmoid
        elif self._source_link_func == "leaky_relu":
            self.source_link = lrelu4p
        else:
            raise NotImplementedError(f"{source_link} not in ['exp', 'identity', 'relu', 'leaky_relu', 'sigmoid']")
        if adam:
            self.grad_update = self.adam_grad_update
        else:
            self.grad_update = tf.keras.layers.Lambda(lambda x, t: x)

    def initial_states(self, batch_size):
        x = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, 1))
        states = self.model.init_hidden_states(self.source_pixels, batch_size)
        self._grad_mean = tf.zeros_like(x, dtype=DTYPE)
        self._grad_var = tf.zeros_like(x, dtype=DTYPE)
        return x, states

    def adam_grad_update(self, grad, time_step):
        time_step = tf.cast(time_step, DTYPE)
        self._grad_mean = self.beta_1 * self._grad_mean + (1 - self.beta_1) * grad
        self._grad_var  = self.beta_2 * self._grad_var + (1 - self.beta_2) * tf.square(grad)
        # for grad update, unbias the moments
        m_hat = self._grad_mean / (1 - self.beta_1**(time_step + 1))
        v_hat = self._grad_var / (1 - self.beta_2**(time_step + 1))
        return m_hat / (tf.sqrt(v_hat) + self.epsilon)

    def time_step(self, source, grad, states):
        source, states = self.model(xt=source, states=states, grad=grad)
        return source, states

    def __call__(self, lensed_image, outer_tape=nulltape):
        return self.call(lensed_image, outer_tape)

    def call(self, lensed_image, kappa, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa)
                    cost = tf.reduce_mean(log_likelihood)
            grad = g.gradient(cost, source)
            grad = self.grad_update(grad, current_step)
            source, states = self.time_step(source, grad, states)
            source_series = source_series.write(index=current_step, value=source)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa)
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return source_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, kappa, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa)
                    cost = tf.reduce_mean(log_likelihood)
            grad = g.gradient(cost, source)
            grad = self.grad_update(grad, current_step)
            source, states = self.time_step(source, grad, states)
            source_series = source_series.write(index=current_step, value=self.source_link(source))
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=log_likelihood)
        # last step score
        log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=self.source_link(source), kappa=kappa)
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=log_likelihood)
        return source_series.stack(), chi_squared_series.stack()
