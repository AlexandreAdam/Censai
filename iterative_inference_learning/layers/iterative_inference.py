import tensorflow as tf
from iterative_inference_learning.layers.rnn_cell import GRUCellFlex, MultiRNNCellFlex
from iterative_inference_learning.layers.rnn import flex_rnn
from iterative_inference_learning.layers import loopfun
from iterative_inference_learning.layers.utils import NameWrapper

DEFAULT_FUNC = lambda _x, _n : tf.contrib.layers.conv2d(_x, _n, [3,3], activation_fn=None)
DEFAULT_T= tf.constant(-1, dtype=tf.int32)
DEFAULT_LINK = tf.nn.sigmoid

def function(x_init, error_fun, reconstruction_loss, n_channel=3, rank=4,
             T=DEFAULT_T, output_specs=None, func=DEFAULT_FUNC, features = [64,64], rnn=GRUCellFlex,
             p_prior=0.25, t_max=150, n_pseudo=2, accumulate_output=True, init_func=DEFAULT_FUNC):

    batch_size = tf.shape(x_init)[0]
    split_dim = rank - 1

    # Sample T if necessary
    T = tf.cond(T < 0, lambda: _sample_T(p_prior, t_max), lambda: T)
    # Set the max time for loopfun
    max_time = tf.ones([batch_size], dtype=tf.int32) * T
    # Get prior probabilities
    p_t = _get_p_t_given_T(p_prior, T, batch_size)

    # Specifiy the output names for later usage
    output_specs.update({'pseudo_samples': n_pseudo*n_channel})
    output_specs.update({'resp': 1})
    names = _output_specs_toname(output_specs, split_dim)

    # Define RNN cell
    cell = MultiRNNCellFlex([rnn(n_features, tensor_rank=rank, function=func, inner_function=func)
                             for n_features in features], state_is_tuple=False)

                             
    # Input function, this function generates the input that is fed into cell
    ifunc = loopfun.InputFunction(split_dim, [], [],
                                  [names.name_dict['all'], names.name_dict['pseudo_samples']],
#                                  [tf.identity, loopfun.ApplyMultFunction(loopfun.GradientFunction(error_fun), split_dim, n_pseudo)])
                                  [tf.identity, loopfun.ApplyMultFunction(loopfun.GradientFunction(error_fun), split_dim)])
    # Output transform function: transforms and accumulates output of cell over time
    ofunc = loopfun.OutputFunction(lambda _x: func(_x,names.num_dim), accum=accumulate_output)
    # This function returns a bool whether max time has been reached
    sfunc = loopfun.StoppingFunction(max_time)
    loop_fn = loopfun.LoopFunction(ifunc, ofunc, sfunc)

    # initialisation function which produces the initial input and state
    def init_fn(time, cell):
      new_output = init_func(x_init,names.num_dim)
      new_input = ifunc(x_init, new_output)
      new_state = cell.init_state(tf.unstack(tf.shape(new_input)[:split_dim]), tf.float32)
      elements_finished = sfunc(time)

      return (elements_finished, new_input, new_output, new_state)
    
    # debugging:
    print init_fn, "debug"
    
    # Iterate with the RNN
    outputs_ta, final_state = flex_rnn(cell, loop_fn, init_fn, swap_memory=True)

    # Turn TensorArray into tensor
    alltime_output = tf.TensorArray.pack(outputs_ta)

    # Calculate posterior probabilities over time steps
    # First extract resp slices
    alltime_resp = names(alltime_output, 'resp', split_dim=split_dim + 1)
    # Then average over all feature dimensions (not over batch and time), transpose necessary for softmax
    alltime_log_resp = tf.transpose(tf.reduce_mean(alltime_resp, reduction_indices=range(-rank+1,0)))
    # Apply softmax
    alltime_resp = tf.nn.softmax(alltime_log_resp)

    # KL divergence term for the posterior
    KL_div = tf.reduce_mean(tf.reduce_sum(
        alltime_resp * (tf.nn.log_softmax(alltime_log_resp) - tf.log(tf.transpose(p_t))),
        reduction_indices=[1]))

    # Transpose back for multiplication with reconstruction error
    alltime_resp = tf.transpose(alltime_resp)

    loss = tf.reduce_mean(tf.reduce_sum(
      alltime_resp*reconstruction_loss(alltime_output, names), reduction_indices=[0])) + KL_div

    return T, loss, KL_div, alltime_resp, alltime_output, names

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

def _output_specs_toname(output_specs, split_dim):
    temp_dict = {}
    running_idx = 0

    for name, val in output_specs.items():
        temp_dict.update({name:[running_idx,val]})
        running_idx = running_idx + val

    return NameWrapper(temp_dict, split_dim)
