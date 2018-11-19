import numpy as np
import tensorflow as tf
from iterative_inference_learning.layers.rnn_cell import BasicRNNCellFlex, GRUCellFlex, MultiRNNCellFlex, EmbeddingWrapperFlex, FakeRNNCellFlex, OutputProjectionWrapperFlex
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.training import moving_averages

def batch_normalization(incoming, beta_init=0.0, gamma_init=1.0, epsilon=1e-5,
                        decay=0.9, is_training=True, trainable=True,
                        restore=True, reuse=False, scope=None,
                        name="BatchNormalization"):
    """ Batch Normalization.
    Normalize activations of the previous layer at each batch.
    Arguments:
        incoming: `Tensor`. Incoming Tensor.
        beta: `float`. Default: 0.0.
        gamma: `float`. Default: 1.0.
        epsilon: `float`. Defalut: 1e-5.
        decay: `float`. Default: 0.9.
        stddev: `float`. Standard deviation for weights initialization.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).
    References:
        Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shif. Sergey Ioffe, Christian Szegedy. 2015.
    Links:
        [http://arxiv.org/pdf/1502.03167v3.pdf](http://arxiv.org/pdf/1502.03167v3.pdf)
    """

    input_shape = incoming.get_shape()
    input_ndim = len(input_shape)

    # Variable Scope fix for older TF
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    with vscope:
        beta = tf.get_variable('beta', shape=[input_shape[-1]],
                           initializer=tf.constant_initializer(beta_init),
                           trainable=trainable)
        gamma = tf.get_variable('gamma', shape=[input_shape[-1]],
                            initializer=tf.constant_initializer(np.log(gamma_init)), trainable=trainable)

        axis = list(range(input_ndim - 1))
        moving_mean = tf.get_variable('moving_mean',
                                  input_shape[-1:],
                                  initializer=tf.constant_initializer(beta_init),
                                  trainable=False)
        moving_variance = tf.get_variable('moving_variance',
                                      input_shape[-1:],
                                      initializer=tf.constant_initializer(gamma_init),
                                      trainable=False)

        # Define a function to update mean and variance
        def update_mean_var():
            mean, variance = tf.nn.moments(incoming, axis)

            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, mean, decay)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, variance, decay)
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                #return tf.identity(mean), tf.identity(variance)
                return moving_mean, moving_variance

        mean, var = tf.cond(
            is_training, update_mean_var, lambda: (moving_mean, moving_variance))

        try:
            inference = tf.nn.batch_normalization(
                incoming, mean, var, beta, tf.exp(gamma), epsilon)
            inference.set_shape(input_shape)
        # Fix for old Tensorflow
        except Exception as e:
            inference = tf.nn.batch_norm_with_global_normalization(
                incoming, mean, var, beta, tf.exp(gamma), epsilon,
                scale_after_normalization=True,
            )
            inference.set_shape(input_shape)

    return inference


def gru(k_size, features, is_training):

    linear_func, linear_input, normalizer, output_func = _make_network(k_size, is_training)

    rnn = lambda n_features: GRUCellFlex(n_features, tensor_rank=4, function=linear_func, inner_function=linear_func)
    #rnn = lambda n_features: BasicRNNCellFlex(n_features, tensor_rank=4, function=linear_func,
    #                                          activation=tf.nn.elu)

    rnns = [rnn(n_features) for n_features in features[1:]]

    rnns[0] = EmbeddingWrapperFlex(rnns[0], linear_input, features[0], normalizer)
    #rnns[-1] = OutputProjectionWrapperFlex(rnns[-1], features[-1], linear_func, normalizer)

    cell = MultiRNNCellFlex(rnns, state_is_tuple=True)

    return cell, output_func

def _make_network(k_size, is_training):
    normalizer = lambda x_: tf.identity(x_)

    # batch_norm = lambda x_: tf.contrib.slim.batch_norm(x_, decay = 0.9, center=True, scale=True,
    #                                                    is_training=is_training,
    #                                                    activation_fn=None)
    batch_norm = lambda x_: batch_normalization(x_, decay=0.99, is_training=is_training,
                                                trainable=True)

    linear_func = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size,k_size],
                                                              activation_fn=None, padding="SAME",
                                                              normalizer_fn=None,#batch_norm,
                                                              rate=2, weights_regularizer=l2_regularizer(1e-5))

    linear_input = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size+2,k_size+2],
                                                                  activation_fn=tf.nn.tanh, padding="SAME",
                                                                  normalizer_fn=None,#batch_norm,
                                                                  weights_regularizer=l2_regularizer(1e-5))#,
                                                                  #biases_initializer=None)

    output_func = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size,k_size],
                                                                  activation_fn=None, padding="SAME",
                                                                  weights_regularizer=l2_regularizer(1e-5),
                                                                  biases_initializer=None)


    return linear_func, linear_input, normalizer, output_func