import tensorflow as tf
from .definitions import lrelu4p
from .models.rim_unet_model import UnetModel


class RIM_UNET_CELL(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self, physical_model, batch_size, num_steps, num_pixels, input_size=None, activation=tf.tanh,
                 cond1=None, cond2=None, **kwargs):
        super(RIM_UNET_CELL, self).__init__(**kwargs)
        self.physical_model = physical_model
        self.num_pixels = num_pixels  # state size is not used, fixed in self.state_size_list
        self.num_steps = num_steps
        self._num_units = 32
        # self.double_RIM_state_size = state_size #Not used
        # self.single_RIM_state_size = state_size//2 # Not used
        # self.gru_state_size = state_size//4
        # self.gru_state_pixel_downsampled = 16*2
        self._activation = activation
        self.state_size_list = [32, 32, 128, 512]
        self.model_1 = UnetModel(self.state_size_list)
        self.model_2 = UnetModel(self.state_size_list)
        self.batch_size = batch_size
        self.initial_condition(cond1, cond2)
        self.initial_output_state()

    def initial_condition(self, cond1, cond2):
        if cond1 is None:
            self.inputs_1 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, 1))
        else:
            self.inputs_1 = tf.identity(cond1)
        if cond2 is None:
            self.inputs_2 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, 1))
        else:
            self.inputs_2 = tf.identity(cond2)

    def initial_output_state(self):
        STRIDE = self.model_1.STRIDE
        numfeat_1, numfeat_2, numfeat_3, numfeat_4 = self.state_size_list
        state_11 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, numfeat_1))
        state_12 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 1, self.num_pixels // STRIDE ** 1, numfeat_2))
        state_13 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 2, self.num_pixels // STRIDE ** 2, numfeat_3))
        state_14 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 3, self.num_pixels // STRIDE ** 3, numfeat_4))
        self.state_1 = [state_11, state_12, state_13, state_14]

        STRIDE = self.model_2.STRIDE
        state_21 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, numfeat_1))
        state_22 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 1, self.num_pixels // STRIDE ** 1, numfeat_2))
        state_23 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 2, self.num_pixels // STRIDE ** 2, numfeat_3))
        state_24 = tf.zeros(
            shape=(self.batch_size, self.num_pixels // STRIDE ** 3, self.num_pixels // STRIDE ** 3, numfeat_4))
        self.state_2 = [state_21, state_22, state_23, state_24]

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2, scope=None):
        xt_1, ht_1 = self.model_1(inputs_1, state_1, grad_1)
        xt_2, ht_2 = self.model_2(inputs_2, state_2, grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def forward_pass(self, data):
        # 1=source, 2=kappa
        output_series_1 = []
        output_series_2 = []
        with tf.GradientTape() as g:
            g.watch(self.inputs_1)
            g.watch(self.inputs_2)
            cost = self.physical_model.log_likelihood(y_true=data, source=self.inputs_1, kappa=self.inputs_2)
        grads = g.gradient(cost, [self.inputs_1, self.inputs_2])

        output_1, state_1, output_2, state_2 = self.__call__(self.inputs_1, self.state_1, grads[0], self.inputs_2,
                                                             self.state_2, grads[1])
        output_series_1.append(output_1)
        output_series_2.append(output_2)

        for current_step in range(self.num_steps - 1):
            with tf.GradientTape() as g:
                g.watch(output_1)
                g.watch(output_2)
                cost = self.physical_model.log_likelihood(y_true=data, source=output_1, kappa=output_2)
            grads = g.gradient(cost, [output_1, output_2])
            output_1, state_1, output_2, state_2 = self.__call__(output_1, state_1, grads[0], output_2, state_2,
                                                                 grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)
        return output_series_1, output_series_2, cost

    def cost_function(self, data, source, kappa):
        output_series_1, output_series_2, final_log_L = self.forward_pass(data)
        chi1 = sum([tf.square(output_series_1[i] - source) for i in range(self.num_steps)]) / self.num_steps
        chi2 = sum([tf.square(output_series_2[i] - kappa) for i in range(self.num_steps)]) / self.num_steps
        return tf.reduce_mean(chi1) + tf.reduce_mean(
            chi2)  # , output_series_1 , output_series_2 , output_series_1[-1].numpy() , output_series_2[-1].numpy()
