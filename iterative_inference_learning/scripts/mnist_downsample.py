'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
import iterative_inference_learning.layers.iterative_inference as iterative_inference

import tflearn.datasets.mnist as mnist

X, Y, testX, testY = mnist.load_data(one_hot=True)

batch_size = 50
sx, sy, n_channel = 28, 28, 1
input_depth = sx * sy * n_channel

def unpooling2x2(x):
    out = tf.concat([x,x],3)
    out = tf.concat([out, out],2)

    sh = x.get_shape().as_list()
    if None not in sh[1:]:
        out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
        return tf.reshape(out, out_size)
    else:
        sh = tf.shape(x)
        return tf.reshape(out, [-1, sh[1] * 2, sh[2] * 2, sh[3]])

def encoder(x):
    return tf.contrib.layers.avg_pool2d(x, 4, stride=4, padding="SAME")

def decoder(x):
    x_dec = unpooling2x2(unpooling2x2(x))
    x_dec = tf.minimum(tf.maximum(1e-6, x_dec), 1. - 1e-6)
    x_dec = tf.log(x_dec) - tf.log((1. - x_dec))
    return x_dec

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
T = tf.constant(-1, dtype=tf.int32)
input_data = tf.placeholder(dtype=tf.float32, shape=(None, input_depth), name='input')
data = tf.reshape(input_data,[-1, sx, sy, n_channel])
y = encoder(data)
x_init = decoder(y)

def mse(x_test):
    return tf.reduce_sum(- 0.5 * tf.square(encoder(tf.sigmoid(x_test)) - y))

def lossfun(x_est, names):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        names(x_est, 'mu',4),tf.expand_dims(data,0)), [-3,-2,-1])

T_, loss, KL_div, alltime_resp, alltime_output, names = \
    iterative_inference.function(x_init=x_init, error_fun=mse, reconstruction_loss=lossfun,\
                                 n_channel=n_channel, T=T, output_specs={'mu':n_channel}, p_prior=0.25, t_max=150,n_pseudo=2)

minimize = tf.contrib.layers.optimize_loss(loss, global_step, 0.0001, "Adam", clip_gradients=5.0)

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
        if j % 10 == 0:
            print [i, j] + sess.run([loss, KL_div],{input_data: inp, T:10}) + sess.run([loss, KL_div],{input_data: inp, T:20}) + sess.run([loss, KL_div],{input_data: inp, T:50})
        # print sess.run([t, loss, KL_div],{input_data: inp})#, sess.run(loss,{input_data: inp}), sess.run(KL_div,{input_data: inp})
        # print sess.run(t,{input_data: inp})
        # print sess.run(prior_resp,{input_data: inp}).shape
    print "Epoch - ",str(i)
sess.close()