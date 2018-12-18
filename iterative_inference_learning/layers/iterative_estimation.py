import tensorflow as tf
from iterative_inference_learning.layers.rnn import flex_rnn
from iterative_inference_learning.layers import loopfun


def function(x_init1, x_init2, cell1, cell2, input_func1, input_func2, output_func1, output_func2, init_func1, init_func2,
             T=tf.constant(-1, dtype=tf.int32), p_prior=0.1, t_max=150):

    input_shape1 = tf.shape(x_init1)
    input_shape2 = tf.shape(x_init2)
    batch_size, sx, sy = input_shape[0], input_shape[1], input_shape[2]

    # Sample T if necessary
    t = tf.cond(T < 0, lambda: _sample_T(p_prior, t_max), lambda: T)
    # Set the max time for loopfun
    max_time = tf.ones([batch_size], dtype=tf.int32) * t
    # Get prior probabilities
    p_t = tf.cond(T < 0, lambda: _get_p_t_given_T(p_prior, t, batch_size),
                         lambda: tf.ones([T,batch_size]) / tf.to_float(T))

    # This function returns a bool whether max time has been reached
    stopping_func = loopfun.StoppingFunction(max_time)
    loop_fn1 = loopfun.LoopFunction(input_func1, output_func1, stopping_func)
    loop_fn2 = loopfun.LoopFunction(input_func2, output_func2, stopping_func)

     # Iterate with the RNN
    outputs_ta1, outputs_ta2 , final_state1 , final_state2 = flex_rnn(cell1, cell2 , loop_fn1, loop_fn2, init_func1(x_init1, input_shape1, stopping_func), init_func2(x_init2, input_shape2, stopping_func), swap_memory=False)

    # Turn TensorArray into tensor
    alltime_output1 = tf.TensorArray.stack(outputs_ta1)
    final_output1 = outputs_ta1.read(t-1)
    alltime_output2 = tf.TensorArray.stack(outputs_ta2)
    final_output2 = outputs_ta2.read(t-1)

    return alltime_output1, alltime_output2, final_output1, final_output2, final_state1, final_state2, p_t, t

def _pdf_T(p, t_max):
    q = 1. - p
    q = tf.sqrt(q)
    T = tf.cumsum(tf.ones([1,t_max]),1)
    pdf = tf.pow(q, T - 1.) * (1. - tf.pow(q, T)) * (1. - tf.square(q))

    return pdf

def _sample_T(p, t_max):
    pdf = _pdf_T(p, t_max)
    pdf = pdf/tf.reduce_sum(pdf)
    t = tf.to_int32(tf.reduce_sum(tf.multinomial(tf.log(pdf), 1))) + 1

    return t

def _get_p_t_given_T(p, T, batch_size):
    q = 1. - p
    q = tf.sqrt(q)
    p_t = tf.pow(q, tf.cumsum(tf.ones([T]),0) - 1.)
    p_t = p_t * (1. - q) / (1. - tf.pow(q, tf.to_float(T)))
    p_t = tf.tile(tf.expand_dims(p_t,1), [1,batch_size])

    return p_t