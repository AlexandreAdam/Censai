'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import layers.iterative_inference as iterative_inference
import layers.iterative_estimation as iterative_estimation
import layers.loopfun as loopfun
from layers.rnn_cell import GRUCellFlex, MultiRNNCellFlex, OutputProjectionWrapperFlex

import tflearn.datasets.mnist as mnist

linear_func = lambda _x, _n : tf.contrib.layers.conv2d(_x, _n, [3,3], activation_fn=None)
features = [64, 64]

X, Y, testX, testY = mnist.load_data(one_hot=True)

batch_size = 20
sx, sy, n_channel = 28, 28, 1
input_depth = sx * sy * n_channel

K = tf.random_uniform([1], maxval=input_depth, dtype=tf.int32)
rp_dims = tf.reduce_sum(tf.stack([K,  tf.expand_dims(tf.constant(input_depth, dtype=tf.int32),0)]), [1])
noise_var = tf.reduce_sum(tf.random_uniform([1], maxval=1., dtype=tf.float32))#tf.random_gamma([1], .1))

def gaussian_rp():
    return tf.random_normal(rp_dims)

def binary_rp():
     return tf.where(tf.random_uniform(rp_dims) > 0.5, tf.ones(rp_dims), -tf.ones(rp_dims))

def fourier_rp():
    return tf.random_uniform(rp_dims) > 0.5

R = 1./tf.sqrt(tf.to_float(K)) * tf.cond(tf.reduce_sum(tf.random_uniform([1])) <= 0.5, gaussian_rp, binary_rp)

def encoder(x):
    return tf.matmul(tf.reshape(x, (batch_size,-1)), R) #adj_y=True)

def decoder(x):
    x = tf.matmul(tf.reshape(x, (batch_size,-1)), R)
    return tf.reshape(x, (batch_size, sx, sy, n_channel))

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
T = tf.constant(20, dtype=tf.int32)
input_data = tf.placeholder(dtype=tf.float32, shape=(None, input_depth), name='input')
data = tf.reshape(input_data,[-1, sx, sy, n_channel])
y = encoder(data)
y = y + tf.random_normal(tf.shape(y), stddev=tf.sqrt(noise_var))
x_init = decoder(y)

def param2image(x_param):
    x_temp = tf.nn.softplus(x_param) / 255.
    x_temp = tf.minimum(x_temp, tf.constant(1, dtype=tf.float32))
    # x_temp = x_temp/tf.reduce_mean(x_temp,[-3,-2], True)
    return x_temp

def mse(x_test):
    return tf.reduce_sum(- 0.5 / tf.sqrt(noise_var) * tf.square(encoder(param2image(x_test)) - y))

def lossfun(x_est, expand_dim = False):
    temp_data = data
    if expand_dim:
        temp_data  = tf.expand_dims(temp_data,0)
    return tf.reduce_sum(0.5 * tf.square(param2image(x_est) - temp_data), [-3,-2,-1])

def log10(x):
    return tf.log(x)/tf.log(tf.constant(10, dtype=tf.float32))

# Define RNN cell
cell = MultiRNNCellFlex([GRUCellFlex(n_features, tensor_rank=4, function=linear_func, inner_function=linear_func)
                             for n_features in features], state_is_tuple=False)
# cell = OutputProjectionWrapperFlex(cell, n_channel, linear_func)

alltime_output, final_output, finale_state, p_t, T_, names = \
    iterative_estimation.function(x_init, cell, loopfun.GradientFunction(mse), n_channel=1, rank=4,
                                  output_specs={'mu':n_channel}, ofunc=linear_func, T=T, accumulate_output=True)
alltime_output = names(alltime_output, 'mu', 4)
final_output = names(final_output, 'mu')

loss_full = tf.reduce_sum(tf.reduce_mean(p_t * lossfun(alltime_output, True), reduction_indices=[1]))
loss = tf.reduce_mean(lossfun(final_output))
psnr = tf.reduce_mean(20. * log10(tf.constant(255, dtype=tf.float32))
                      - 10. * log10(tf.reduce_mean(tf.square(255.*(data - param2image(final_output))), [-3,-2,-1])))
psnr_x_init = tf.reduce_mean(20. * log10(tf.constant(255, dtype=tf.float32))
                      - 10. * log10(tf.reduce_mean(tf.square(255.*(data - x_init)), [-3,-2,-1])))
minimize = tf.contrib.layers.optimize_loss(loss_full, global_step, 0.0001, "Adam", clip_gradients=5.0)

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

no_of_batches = int(len(X)/batch_size)
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = X[ptr:ptr+batch_size], Y[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run([minimize],{input_data: inp})
        if j % 50 == 0:
            proj_mat, data_corrupt, error_var, k_sparse = sess.run([R, y, noise_var, K],{input_data: inp, T:-1})
            print [i, j, k_sparse[0], 255.*error_var] + sess.run([loss, psnr_x_init, T_],{input_data: inp, T:1, R:proj_mat, y:data_corrupt, noise_var:error_var}) \
                  + sess.run([loss, psnr, T_],{input_data: inp, T:10, R:proj_mat, y:data_corrupt, noise_var:error_var}) \
                  + sess.run([loss, psnr, T_],{input_data: inp, T:20, R:proj_mat, y:data_corrupt, noise_var:error_var}) \
                  + sess.run([loss, psnr, T_],{input_data: inp, T:50, R:proj_mat, y:data_corrupt, noise_var:error_var})
        # print sess.run([t, loss, KL_div],{input_data: inp})#, sess.run(loss,{input_data: inp}), sess.run(KL_div,{input_data: inp})
        # print sess.run(t,{input_data: inp})
        # print sess.run(prior_resp,{input_data: inp}).shape
    print "Epoch - ",str(i)
sess.close()