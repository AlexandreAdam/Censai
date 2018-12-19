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


def flex_rnn(cell1, cell2, loop_fn1, loop_fn2, init_fn1=None, init_fn2=None, dtype=tf.float32,
            parallel_iterations=10, swap_memory=False, scope=None):

  if not isinstance(cell1, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not callable(loop_fn1):
    raise TypeError("loop_fn must be a callable")

  if init_fn1 is None:
    init_fn1 = loop_fn1

  if init_fn2 is None:
    init_fn2 = loop_fn2

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "RNN") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)

    # Perform first time step
    time = constant_op.constant(0, dtype=dtypes.int32)

    (elements_finished1, new_input1, current_output1, current_state1) = init_fn1(time, cell1 , scope='blockA')
    (elements_finished2, new_input2, current_output2, current_state2) = init_fn2(time, cell2 , scope='blockB')

    # Gather some information about the RNN Output
    flat_output_structure1 = nest.flatten(current_output1)
    flat_output_dtypes1 = [emit.dtype for emit in flat_output_structure1]
    flat_output_structure2 = nest.flatten(current_output2)
    flat_output_dtypes2 = [emit.dtype for emit in flat_output_structure2]

    # Initialise List TensorArrays for output
    # First initialise them flat
    flat_output_ta1 = [
        tensor_array_ops.TensorArray(
            dtype=dtype_i, dynamic_size=True, size=0, clear_after_read=False, name="rnn_output_%d" % i)
        for i, dtype_i in enumerate(flat_output_dtypes1)]
    # Then turn it into a hierarchical form
    output_ta1 = nest.pack_sequence_as(structure=current_output1,
                                    flat_sequence=flat_output_ta1)

    flat_output_ta2 = [
        tensor_array_ops.TensorArray(
            dtype=dtype_i, dynamic_size=True, size=0, clear_after_read=False, name="rnn_output_%d" % i)
        for i, dtype_i in enumerate(flat_output_dtypes2)]
    # Then turn it into a hierarchical form
    output_ta2 = nest.pack_sequence_as(structure=current_output2,
                                    flat_sequence=flat_output_ta2)

    # This function track the stopping criterion
    # One all elements (i.e. batch iterations) are finished, the while loop will stop
    def condition(_unused_time, elements_finished1, *_):
      return math_ops.logical_not(math_ops.reduce_all(elements_finished1))

    def body(time, elements_finished1, elements_finished2, current_input1, current_input2, old_output1, old_output2, old_state1, old_state2, output_ta1,output_ta2):
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
      (next_finished1, new_input1, new_output1, new_state1) = loop_fn1(
         current_time, cell1, current_input1, old_output1, old_state1, old_output2)

      (next_finished2, new_input2, new_output2, new_state2) = loop_fn2(
         current_time, cell2, current_input2, old_output2, old_state2, old_output1)

      nest.assert_same_structure(old_state1, new_state1)
      nest.assert_same_structure(old_output1, new_output1)
      nest.assert_same_structure(current_input1, new_input1)
      nest.assert_same_structure(output_ta1, new_output1)


      def _copy_some_through(current, candidate,elements_finished):
        current_flat = nest.flatten(current)
        candidate_flat = nest.flatten(candidate)
        result_flat = [
            tf.where(elements_finished, current_i, candidate_i)
            for (current_i, candidate_i) in zip(current_flat, candidate_flat)]
        return nest.pack_sequence_as(
            structure=current, flat_sequence=result_flat)

      current_output1 = _copy_some_through(old_output1, new_output1,elements_finished1)
      current_state1 = _copy_some_through(old_state1, new_state1,elements_finished1)
      current_output2 = _copy_some_through(old_output2, new_output2,elements_finished2)
      current_state2 = _copy_some_through(old_state2, new_state2,elements_finished2)

      current_output_flat1 = nest.flatten(current_output1)
      output_ta_flat1 = nest.flatten(output_ta1)
      current_output_flat2 = nest.flatten(current_output2)
      output_ta_flat2 = nest.flatten(output_ta2)

      elements_finished1 = math_ops.logical_or(elements_finished1, next_finished1)
      elements_finished2 = math_ops.logical_or(elements_finished2, next_finished2)

      output_ta_flat1 = [
          ta.write(time, emit)
          for (ta, emit) in zip(output_ta_flat1,current_output_flat1)]
      output_ta_flat2 = [
          ta.write(time, emit)
          for (ta, emit) in zip(output_ta_flat2,current_output_flat2)]


      output_ta1 = nest.pack_sequence_as(
          structure=output_ta1, flat_sequence=output_ta_flat1)
      output_ta2 = nest.pack_sequence_as(
          structure=output_ta2, flat_sequence=output_ta_flat2)


      return (current_time, elements_finished1, elements_finished2, new_input1, new_input2,
              current_output1,current_output2, current_state1, current_state2, output_ta1, output_ta2)

    returned = control_flow_ops.while_loop(
        condition, body, loop_vars=[time, elements_finished1, elements_finished2, new_input1, new_input2,
                                    current_output1,current_output2, current_state1,current_state2, output_ta1,output_ta2],
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    (final_state1,final_state2, output_ta1, output_ta2) = returned[-4:]

    return (output_ta1, output_ta2 , final_state1 , final_state2)
