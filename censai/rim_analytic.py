import tensorflow as tf
from censai.models import ModelAnalytic
from censai.definitions import DTYPE
from censai import AnalyticalPhysicalModelv2
from censai.utils import nulltape


class RIMAnalytic:
    def __init__(
            self,
            physical_model: AnalyticalPhysicalModelv2,
            model: ModelAnalytic,
            steps: int,
            adam=True,
            beta_1=0.9,
            beta_2=0.99,
            epsilon=1e-8
    ):
        self.physical_model = physical_model
        self.steps = steps
        self.model = model
        self.adam = adam
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        if adam:
            self.grad_update = self.adam_grad_update
        else:
            self.grad_update = lambda x, t: x

        self.link = tf.keras.layers.Lambda(lambda x: self.physical_model.model_to_physical(x))
        self.inverse_link = tf.keras.layers.Lambda(lambda x: self.physical_model.physical_to_model(x))

    def initial_states(self, batch_size):
        x = tf.zeros(shape=(batch_size, 13)) + 1e-2
        states = self.model.init_hidden_states(batch_size)
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

    def time_step(self, xt, grad, states):
        xt_augmented = tf.constant([1., 1e-3, 1e-3, 0., 0., 1e-3, 1e-3, 0., 0., 1e-3, 1e-3, 1., 1.], DTYPE)[tf.newaxis, :]  # r_ein, e1, e2, x0, y0, gamma1, gamma2, xs, ys, e1s, e2s, n, r_eff
        dxt, states = self.model(xt=xt_augmented, states=states)
        xt = xt + dxt
        return xt, states

    def __call__(self, lensed_image, noise_rms, psf_fwhm, outer_tape=nulltape):
        return self.call(lensed_image, noise_rms, psf_fwhm, outer_tape)

    def call(self, lensed_image, noise_rms, psf_fwhm, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        xt, states = self.initial_states(batch_size)

        xt_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(xt)
                    zt = self.link(xt)
                    y_pred = self.physical_model.lens_source_sersic_func_vec(zt, psf_fwhm)
                    log_likelihood = 0.5 * tf.reduce_sum(tf.square(lensed_image - y_pred) / noise_rms[:, None, None, None]**2, axis=(1, 2, 3))
            grad = g.gradient(log_likelihood, xt)
            grad = self.grad_update(grad, current_step)
            xt, states = self.time_step(xt, grad, states)
            xt_series = xt_series.write(index=current_step, value=xt)
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=2*log_likelihood/self.physical_model.pixels**2)
        # last step score
        zt = self.link(xt)
        y_pred = self.physical_model.lens_source_sersic_func_vec(zt, psf_fwhm)
        log_likelihood = 0.5 * tf.reduce_sum(tf.square(lensed_image - y_pred) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=2*log_likelihood/self.physical_model.pixels**2)
        return xt_series.stack(), chi_squared_series.stack()

    def predict(self, lensed_image, noise_rms, psf_fwhm, outer_tape=nulltape):
        batch_size = lensed_image.shape[0]
        xt, states = self.initial_states(batch_size)

        xt_series = tf.TensorArray(DTYPE, size=self.steps)
        chi_squared_series = tf.TensorArray(DTYPE, size=self.steps)
        for current_step in range(self.steps):
            with outer_tape.stop_recording():
                with tf.GradientTape() as g:
                    g.watch(xt)
                    zt = self.link(xt)
                    y_pred = self.physical_model.lens_source_sersic_func_vec(zt, psf_fwhm)
                    log_likelihood = 0.5 * tf.reduce_sum(tf.square(lensed_image - y_pred) / noise_rms[:, None, None, None]**2, axis=(1, 2, 3))
            grad = g.gradient(log_likelihood, xt)
            grad = self.grad_update(grad, current_step)
            xt, states = self.time_step(xt, grad, states)
            xt_series = xt_series.write(index=current_step, value=self.link(xt))
            if current_step > 0:
                chi_squared_series = chi_squared_series.write(index=current_step - 1, value=2*log_likelihood/self.physical_model.pixels**2)
        # last step score
        zt = self.link(xt)
        y_pred = self.physical_model.lens_source_sersic_func_vec(zt, psf_fwhm)
        log_likelihood = 0.5 * tf.reduce_sum(tf.square(lensed_image - y_pred) / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))
        chi_squared_series = chi_squared_series.write(index=self.steps - 1, value=2*log_likelihood/self.physical_model.pixels**2)
        return xt_series.stack(), chi_squared_series.stack()


if __name__ == '__main__':
    phys = AnalyticalPhysicalModelv2()
    lens = phys.lens_source_sersic_func()
    model = ModelAnalytic()
    rim = RIMAnalytic(phys, model, steps=2)
    xt, c = rim.call(lens, noise_rms=tf.constant([0.01], dtype=tf.float32), psf_fwhm=tf.constant([0.1], dtype=tf.float32))
    print(xt.numpy())
    print(c.numpy())
