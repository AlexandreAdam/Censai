import tensorflow as tf
from censai.models import Model
from censai.definitions import DTYPE
from censai import PhysicalModel
from censai.utils import nulltape


class RIMv0:
    def __init__(
            self,
            physical_model: PhysicalModel,
            model: Model,
            steps: int,
            adam=True,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
    ):
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.kappa_pixels
        self.source_pixels = physical_model.src_pixels
        self.steps = steps
        self.source_model = model
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def initial_states(self, batch_size):
        # Define initial guess in physical space, then apply inverse link function to bring them in prediction space
        source_init = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, 1), dtype=DTYPE)

        source_states = self.source_model.init_hidden_states(self.source_pixels, batch_size)
        return source_init, source_states

    def time_step(self, sources, source_states, source_grad):
        new_source, new_source_states = self.source_model(sources, source_states, source_grad)
        return new_source, new_source_states

    def __call__(self, lensed_image, kappa, outer_tape=nulltape):
        return self.call(lensed_image, kappa, outer_tape)

    def call(self, lensed_image, kappa, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        source, source_states = self.initial_states(batch_size)

        source_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(source)
                    log_likelihood = self.physical_model.log_likelihood(y_true=lensed_image, source=source, kappa=kappa)
                    cost = tf.reduce_mean(log_likelihood)
            source_grad, kappa_grad = g.gradient(cost, [source, kappa])
            source, source_states = self.time_step(source, source_states, source_grad)
            source_series = source_series.write(index=current_step, value=source)
            chi_squared_series = chi_squared_series.write(index=current_step, value=log_likelihood)
        return source_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, kappa):
        return self.call(lensed_image, kappa)
