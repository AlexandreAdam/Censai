# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


from tensorflow.python.framework import constant_op, dtypes, ops,tensor_shape
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
import tensorflow as tf


# pylint: disable=protected-access
#_state_size_with_prefix = rnn_cell._state_size_with_prefix
# pylint: enable=protected-access


# WARREN HACK:  WHY NOT JUST PUT THE PROTECTED FUNCTION HERE
def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.
  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.
  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.
  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size
# END WARREN HACK


def flex_rnn(cell, loop_fn, init_fn=None, dtype=tf.float32,
            parallel_iterations=10, swap_memory=False, scope=None):

  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not callable(loop_fn):
    raise TypeError("loop_fn must be a callable")

  if init_fn is None:
    init_fn = loop_fn

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "RNN") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Perform first time step
    time = constant_op.constant(0, dtype=dtypes.int32)
    (elements_finished, new_input, current_output, current_state) = init_fn(time, cell)

    # Gather some information about the RNN Output
    flat_output_structure = nest.flatten(current_output)
    flat_output_dtypes = [emit.dtype for emit in flat_output_structure]

    # Initialise List TensorArrays for output
    # First initialise them flat
    flat_output_ta = [
        tensor_array_ops.TensorArray(
            dtype=dtype_i, dynamic_size=True, size=0, clear_after_read=False, name="rnn_output_%d" % i)
        for i, dtype_i in enumerate(flat_output_dtypes)]
    # Then turn it into a hierarchical form
    output_ta = nest.pack_sequence_as(structure=current_output,
                                    flat_sequence=flat_output_ta)

    # This function track the stopping criterion
    # One all elements (i.e. batch iterations) are finished, the while loop will stop
    def condition(_unused_time, elements_finished, *_):
      return math_ops.logical_not(math_ops.reduce_all(elements_finished))

    def body(time, elements_finished, current_input, old_output, old_state,
             output_ta):
      """Internal while loop body for raw_rnn.

      Args:
        time: time scalar.
        elements_finished: batch-size vector.
        current_input: possibly nested tuple of input tensors.
        emit_ta: possibly nested tuple of output TensorArrays.
        state: possibly nested tuple of state tensors.
        loop_state: possibly nested tuple of loop state tensors.

      Returns:
        Tuple having the same size as Args but with updated values.
      """

      # Perform a new step
      current_time = time + 1
      (next_finished, new_input, new_output, new_state) = loop_fn(
         current_time, cell, current_input, old_output, old_state)

      nest.assert_same_structure(old_state, new_state)
      nest.assert_same_structure(old_output, new_output)
      nest.assert_same_structure(current_input, new_input)
      nest.assert_same_structure(output_ta, new_output)


      def _copy_some_through(current, candidate):
        current_flat = nest.flatten(current)
        candidate_flat = nest.flatten(candidate)
        result_flat = [
            tf.where(elements_finished, current_i, candidate_i)
            for (current_i, candidate_i) in zip(current_flat, candidate_flat)]
        return nest.pack_sequence_as(
            structure=current, flat_sequence=result_flat)

      current_output = _copy_some_through(old_output, new_output)
      current_state = _copy_some_through(old_state, new_state)

      current_output_flat = nest.flatten(current_output)
      output_ta_flat = nest.flatten(output_ta)

      elements_finished = math_ops.logical_or(elements_finished, next_finished)

      output_ta_flat = [
          ta.write(time, emit)
          for (ta, emit) in zip(output_ta_flat,current_output_flat)]

      output_ta = nest.pack_sequence_as(
          structure=output_ta, flat_sequence=output_ta_flat)

      return (current_time, elements_finished, new_input,
              current_output, current_state, output_ta)

    returned = control_flow_ops.while_loop(
        condition, body, loop_vars=[time, elements_finished, new_input,
                                    current_output, current_state, output_ta],
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    (final_state, output_ta) = returned[-2:]

    return (output_ta, final_state)
