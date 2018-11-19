import tensorflow as tf
from iterative_inference_learning.layers.utils import NameWrapper
from iterative_inference_learning.layers import loopfun

def init(rank, output_shape_dict, output_transform_dict, init_name, ofunc = None, accumulate_output=True):
    split_dim = rank -1
    # Specifiy the output ospecs for later usage
    ospecs = NameWrapper.output_specs_toname(output_shape_dict, split_dim)
    keys = output_transform_dict.keys()

    # Input function, this function generates the input that is fed into cell
    input_func = loopfun.InputFunction(split_dim, [], [], [ospecs.name_dict[k_] for k_ in keys],
                                  [loopfun.ApplyMultFunction(output_transform_dict[k_], split_dim) for k_ in keys])

    # Output transform function: transforms and accumulates output of cell over time
    if ofunc is not None:
        output_func = loopfun.OutputFunction(lambda _x: ofunc(_x,ospecs.num_dim), accum=accumulate_output)
    else:
        output_func = loopfun.OutputFunction(tf.identity, accum=accumulate_output)


    # initialisation function which produces the initial input and state
    def init_func(x_init, input_shape, stopping_func):
        def init_fn(time, cell):
          new_output = tf.zeros(tf.unstack(input_shape[:split_dim]) + [ospecs.num_dim])
          new_output = ospecs.insert(new_output, init_name, x_init)
          new_input = input_func(x_init, new_output)
          _, new_state = cell.init_call(new_input)
          elements_finished = stopping_func(time)

          return (elements_finished, new_input, new_output, new_state)

        return init_fn

    return input_func, output_func, init_func, ospecs