import tensorflow as tf
from .models.rim_unet_model512 import UnetModel512
from censai.definitions import logkappa_normalization, log_kappa
from censai import PhysicalModel

LOG10 = tf.math.log(10.)


class RIMUnet512:
    def __init__(
            self,
            physical_model: PhysicalModel,
            steps,
            state_sizes=[4, 32, 128, 512],
            adam=True,
            kappalog=True,
            kappa_normalize=True,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            # checkpoint_manager_source=None,
            # checkpoint_manager_kappa=None,
            **models_kwargs):
        self.physical_model = physical_model
        self.kappa_pixels = physical_model.pixels
        self.source_pixels = physical_model.src_pixels
        self.steps = steps
        self._num_units = 32
        self.state_size_list = state_sizes
        if "source" not in models_kwargs:  # temporary solution
            models_kwargs.update({"source": {"strides": 4}})
        if "kappa" not in models_kwargs:
            models_kwargs.update({"source": {"strides": 4}})
        self.source_model = UnetModel512(self.state_size_list, **models_kwargs["source"])
        self.kappa_model = UnetModel512(self.state_size_list, **models_kwargs["kappa"])
        # if checkpoint_manager_source is not None:
        #     checkpoint_manager_source.checkpoint.restore(checkpoint_manager_source.latest_checkpoint)
        #     print(f"Initialized source model from {checkpoint_manager_source.latest_checkpoint}")
        # if checkpoint_manager_kappa is not None:
        #     checkpoint_manager_kappa.checkpoint.restore(checkpoint_manager_kappa.latest_checkpoint)
        #     print(f"Initialized kappa model from {checkpoint_manager_kappa.latest_checkpoint}")
        self.adam = adam
        self.kappalog = kappalog
        self.kappa_normalize = kappa_normalize
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        if self.kappalog:
            if self.kappa_normalize:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: logkappa_normalization(log_kappa(x), forward=True))
            else:
                self.kappa_link = tf.keras.layers.Lambda(lambda x: log_kappa(x))
        else:
            self.kappa_link = tf.identity

        if self.kappalog:
            if self.kappa_normalize:
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**logkappa_normalization(x, forward=False))
            else:
                self.kappa_inverse_link = tf.keras.layers.Lambda(lambda x: 10**x)
        else:
            self.kappa_inverse_link = tf.identity

    def initial_states(self, batch_size):
        source_init = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, 1))
        if self.kappalog:
            if self.kappa_normalize:
                kappa_init = tf.zeros(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
            else:
                kappa_init = -tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1))
        else:
            kappa_init = tf.ones(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, 1)) / 10

        strides = self.source_model.strides
        numfeat_1, numfeat_2, numfeat_3, numfeat_4 = self.state_size_list
        state_11 = tf.zeros(shape=(batch_size, self.source_pixels, self.source_pixels, numfeat_1))
        state_12 = tf.zeros(
            shape=(batch_size, self.source_pixels // strides ** 1, self.source_pixels // strides ** 1, numfeat_2))
        state_13 = tf.zeros(
            shape=(batch_size, self.source_pixels // strides ** 2, self.source_pixels // strides ** 2, numfeat_3))
        state_14 = tf.zeros(
            shape=(batch_size, self.source_pixels // strides ** 3, self.source_pixels // strides ** 3, numfeat_4))
        state_1 = [state_11, state_12, state_13, state_14]

        strides = self.kappa_model.strides
        state_21 = tf.zeros(shape=(batch_size, self.kappa_pixels, self.kappa_pixels, numfeat_1))
        state_22 = tf.zeros(
            shape=(batch_size, self.kappa_pixels // strides ** 1, self.kappa_pixels // strides ** 1, numfeat_2))
        state_23 = tf.zeros(
            shape=(batch_size, self.kappa_pixels // strides ** 2, self.kappa_pixels // strides ** 2, numfeat_3))
        state_24 = tf.zeros(
            shape=(batch_size, self.kappa_pixels // strides ** 3, self.kappa_pixels // strides ** 3, numfeat_4))
        state_2 = [state_21, state_22, state_23, state_24]
        return source_init, state_1, kappa_init, state_2

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

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

    def __call__(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2, scope=None):
        xt_1, ht_1 = self.source_model(inputs_1, state_1, grad_1)
        xt_2, ht_2 = self.kappa_model(inputs_2, state_2, grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def forward_pass(self, data):
        batch_size = data.shape[0]
        source_init, state_1, kappa_init, state_2 = self.initial_states(batch_size)
        # 1=source, 2=kappa
        output_series_1 = []
        output_series_2 = []
        with tf.GradientTape() as g:
            g.watch(source_init)
            g.watch(kappa_init)
            cost = self.physical_model.log_likelihood(y_true=data, source=source_init, kappa=self.kappa_inverse_link(kappa_init))
        grads = g.gradient(cost, [source_init, kappa_init])
        grads = self.grad_update(*grads, 0)

        output_1, state_1, output_2, state_2 = self.__call__(source_init, state_1, grads[0], kappa_init,
                                                             state_2, grads[1])
        output_series_1.append(output_1)
        output_series_2.append(output_2)

        for current_step in range(1, self.steps):
            with tf.GradientTape() as g:
                g.watch(output_1)
                g.watch(output_2)
                cost = self.physical_model.log_likelihood(y_true=data, source=output_1, kappa=self.kappa_inverse_link(output_2))
            grads = g.gradient(cost, [output_1, output_2])
            grads = self.grad_update(*grads, current_step)
            output_1, state_1, output_2, state_2 = self.__call__(output_1, state_1, grads[0], output_2, state_2,
                                                                 grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)
        return output_series_1, output_series_2, cost

    def cost_function(self, data, source, kappa, reduction=True):
        """

        Args:
            data: Batch of lensed images
            source: Batch of source images
            kappa: Batch of kappa maps
            reduction: Whether or not to reduce the batch dimension in computing the loss or not

        Returns: The average loss over pixels, time steps and (if reduction=True) batch size.

        """
        output_series_1, output_series_2, final_log_L = self.forward_pass(data)
        chi1 = sum([tf.square(output_series_1[i] - source) for i in range(self.steps)]) / self.steps
        chi2 = sum([tf.square(output_series_2[i] - self.kappa_link(kappa)) for i in range(self.steps)]) / self.steps
        if reduction:
            return tf.reduce_mean(chi1) + tf.reduce_mean(chi2)
        else:
            return tf.reduce_mean(chi1, axis=(1, 2, 3)) + tf.reduce_mean(chi2, axis=(1, 2, 3))
