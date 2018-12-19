import tensorflow as tf
from iterative_inference_learning.layers.rnn_cell import GRUCellFlex, MultiRNNCellFlex, EmbeddingWrapperFlex, FakeRNNCellFlex
from tensorflow.contrib.layers import l2_regularizer
from tensorflow.python.ops import variable_scope as vs

def gru(k_size=3, features=[64], is_training=True):

    
    

    linear_func1, linear_pool1, linear_unpool1, normalizer1, output_func1 = _make_network(k_size, is_training)
    cell1 = MultiRNNCellFlex([EmbeddingWrapperFlex(GRUCellFlex(4*n_features, tensor_rank=4, function=linear_func1, inner_function=linear_func1),
                                                  linear_pool1, n_features, normalizer1) for n_features in features]
                            + [EmbeddingWrapperFlex(GRUCellFlex(4*n_features, tensor_rank=4, function=linear_func1, inner_function=linear_func1),
                                                    linear_unpool1, n_features, normalizer1) for n_features in features]
                            , state_is_tuple=True)

    linear_func2, linear_pool2, linear_unpool2, normalizer2, output_func2 = _make_network(k_size, is_training)
    cell2 = MultiRNNCellFlex([EmbeddingWrapperFlex(GRUCellFlex(4*n_features, tensor_rank=4, function=linear_func2, inner_function=linear_func2),
                                                  linear_pool2, n_features, normalizer2) for n_features in features]
                            + [EmbeddingWrapperFlex(GRUCellFlex(4*n_features, tensor_rank=4, function=linear_func2, inner_function=linear_func2),
                                                    linear_unpool2, n_features, normalizer2) for n_features in features]
                            , state_is_tuple=True)

    return cell1 , cell2 , output_func1 , output_func2

# def relu(k_size=3, features=[64], is_training=True):
#
#     linear_func, linear_pool, linear_unpool, normalizer, output_func = _make_network(k_size, is_training)
#
#     cell = MultiRNNCellFlex([EmbeddingWrapperFlex(FakeRNNCellFlex(4*n_features, tensor_rank=4, function=linear_func,
#                                                                   activation=tf.nn.relu),
#                                                   linear_pool, n_features, normalizer) for n_features in features]
#                             + [EmbeddingWrapperFlex(FakeRNNCellFlex(4*n_features, tensor_rank=4, function=linear_func,
#                                                                     activation=tf.nn.relu),
#                                                     linear_unpool, n_features, normalizer) for n_features in features]
#                             , state_is_tuple=True)
#
#     return cell, output_func


def _make_network(k_size, is_training):
    normalizer = lambda x_: tf.contrib.slim.batch_norm(x_, center=True, scale=True, is_training=is_training,
                                                        updates_collections=None,
                                                        activation_fn=tf.nn.tanh, scope='bn')

    linear_func = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size,k_size],
                                                              activation_fn=None, padding="SAME",
                                                              weights_regularizer=l2_regularizer(1e-5))

    linear_pool = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size,k_size],
                                                                  activation_fn=None, stride=2, padding="SAME",
                                                                  normalizer_fn=None,
                                                                  weights_regularizer=l2_regularizer(1e-5),
                                                                  biases_initializer=None)

    linear_unpool = lambda _x, _n : tf.contrib.layers.convolution2d_transpose(_x, _n, [k_size,k_size],
                                                                  activation_fn=None, stride=2, padding="SAME",
                                                                  normalizer_fn=None,
                                                                  weights_regularizer=l2_regularizer(1e-5),
                                                                  biases_initializer=None)

    output_func = lambda _x, _n : tf.contrib.layers.convolution2d(_x, _n, [k_size,k_size],
                                                                  activation_fn=None, padding="SAME",
                                                                  weights_regularizer=l2_regularizer(1e-5),
                                                                  biases_initializer=None)



    return linear_func, linear_pool, linear_unpool, normalizer, output_func