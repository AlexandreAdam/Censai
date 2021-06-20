import tensorflow as tf
from .models.rim_model import Model

# TODO dubug this class
class RIM(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, num_steps, num_pixels, state_size, input_size=None, activation=tf.tanh, **kwargs):
        super(RIM, self).__init__(**kwargs)
        self.num_pixels = num_pixels
        self.num_steps = num_steps
        self._num_units = state_size
        self.double_RIM_state_size = state_size
        self.single_RIM_state_size = state_size//2
        self.gru_state_size = state_size//4
        self.gru_state_pixel_downsampled = 16*2
        self._activation = activation
        self.model_1 = Model(self.single_RIM_state_size)
        self.model_2 = Model(self.single_RIM_state_size)
        self.batch_size = batch_size
        self.initial_output_state()

    def initial_output_state(self):
        self.inputs_1 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, 1))
        self.state_1 = tf.zeros(shape=(self.batch_size,  self.num_pixels//self.gru_state_pixel_downsampled, self.num_pixels//self.gru_state_pixel_downsampled, self.single_RIM_state_size))
        self.inputs_2 = tf.zeros(shape=(self.batch_size, self.num_pixels, self.num_pixels, 1))
        self.state_2 = tf.zeros(shape=(self.batch_size,  self.num_pixels//self.gru_state_pixel_downsampled, self.num_pixels//self.gru_state_pixel_downsampled, self.single_RIM_state_size))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2):
        xt_1, ht_1 = self.model_1(inputs_1, state_1 , grad_1)
        xt_2, ht_2 = self.model_2(inputs_2, state_2 , grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def forward_pass(self, data):
        if (data.shape[0] != self.batch_size):
            self.batch_size = data.shape[0]
            self.initial_output_state()


            output_series_1 = []
            output_series_2 = []

            with tf.GradientTape() as g:
                g.watch(self.inputs_1)
                g.watch(self.inputs_2)
                y = log_likelihood(data, physical_model(self.inputs_1, self.inputs_2), noise_rms) #TODO input log_likelihood and physical model in init
            grads = g.gradient(y, [self.inputs_1 , self.inputs_2])

            output_1, state_1, output_2, state_2 = self.__call__(self.inputs_1, self.state_1 , grads[0] , self.inputs_2 , self.state_2 , grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)

            for current_step in range(self.num_steps-1):
                with tf.GradientTape() as g:
                    g.watch(output_1)
                    g.watch(output_2)
                    y = log_likelihood(data, self.physical_model(output_1,output_2),noise_rms)
                grads = g.gradient(y, [output_1 , output_2])


                output_1, state_1 , output_2 , state_2 = self.__call__(output_1, state_1 , grads[0] , output_2 , state_2 , grads[1])
                output_series_1.append(output_1)
                output_series_2.append(output_2)
            final_log_L = log_likelihood(data, self.physical_model(output_1,output_2), noise_rms)
            return output_series_1 , output_series_2 , final_log_L

    def cost_function(self, data, labels_x_1, labels_x_2):
        output_series_1 , output_series_2 , final_log_L = self.forward_pass(data)
        chi1 = sum([tf.square(output_series_1[i] - labels_x_1) for i in range(self.num_steps)]) / self.num_steps
        chi2 = sum([tf.square(output_series_2[i] - labels_x_2) for i in range(self.num_steps)]) / self.num_steps
        return tf.reduce_mean(chi1) + tf.reduce_mean(chi2)#, output_series_1 , output_series_2 , output_series_1[-1].numpy() , output_series_2[-1].numpy()

