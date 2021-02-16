# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.python.ops import array_ops

from tensorflow.python.framework import tensor_shape
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn as rnn_cell
#from tensorflow.python.ops.rnn_cell import RNNCell, GRUCell, MultiRNNCell, BasicRNNCell
from tensorflow.contrib.rnn import RNNCell, GRUCell, MultiRNNCell, BasicRNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

from tensorflow.python.ops.math_ops import sigmoid, tanh
from tensorflow.contrib.layers import fully_connected


DEFAULT_FUNC = lambda x, n: fully_connected(x, n, activation_fn=None)
DEFAULT_RANK = 2

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

class RNNCellFlex(RNNCell):
  """Abstract object representing an RNN cell.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  tuple of integers, then it results in a tuple of `len(state_size)` state
  matrices, each with the a column size corresponding to values in `state_size`.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by the class `MultiRNNCell`,
  or by calling the `rnn` ops several times. Every `RNNCell` must have the
  properties below and and implement `__call__` with the following signature.
  """

  @property
  def rank(self):
    """rank(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def learn_init_state(self):
      raise NotImplementedError("Abstract method")

  def zero_state(self, prefix, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """

    state_size = self.state_size
    if nest.is_sequence(state_size):
      state_size_flat = nest.flatten(state_size)
      zeros_flat = [
          array_ops.zeros(
              array_ops.stack(_state_size_with_prefix(s, prefix=prefix)),
              dtype=dtype)
          for s in state_size_flat]
      zeros = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=zeros_flat)
    else:
      zeros_size = _state_size_with_prefix(state_size, prefix=prefix)
      zeros = array_ops.zeros(array_ops.stack(zeros_size), dtype=dtype)

    return zeros

  def init_state(self, prefix, dtype):
    if not self.learn_init_state:
      return self.zero_state(prefix, dtype)
    else:
      return self.zero_state(prefix, dtype)

  def init_call(self, inputs, scope=None):
    state = self.init_state(tf.unstack(tf.shape(inputs)[:self.rank-1]), dtype=tf.float32)
    out, _ = self(inputs, state)
    return out, state

class BasicRNNCellFlex(BasicRNNCell, RNNCellFlex):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, learn_init_state=False, tensor_rank=DEFAULT_RANK,
               function= DEFAULT_FUNC, activation=tanh):
    super(BasicRNNCellFlex, self).__init__(num_units, activation=activation)
    self._learn_init_state = learn_init_state
    self._tensor_rank = tensor_rank
    self._function = tf.make_template('output_function', function) #function

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def rank(self):
    return self._tensor_rank

  @property
  def learn_init_state(self):
    return self._learn_init_state

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "SimpleCenn"
      new_h = self._activation(_apply_func([inputs, state], self._tensor_rank,
                                             self._num_units, self._function))

    return new_h, new_h

class FakeRNNCellFlex(BasicRNNCell, RNNCellFlex):
  """This cell ignore"""

  def __init__(self, num_units, learn_init_state=False, tensor_rank=DEFAULT_RANK,
               function= DEFAULT_FUNC, activation=tanh):
    super(FakeRNNCellFlex, self).__init__(num_units, activation=activation)
    self._learn_init_state = learn_init_state
    self._tensor_rank = tensor_rank
    self._function = tf.make_template('output_function', function) #function

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def rank(self):
    return self._tensor_rank

  @property
  def learn_init_state(self):
    return self._learn_init_state

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "SimpleCenn"
      new_h = self._activation(_apply_func([inputs], self._tensor_rank,
                                             self._num_units, self._function))

    return new_h, new_h

class GRUCellFlex(GRUCell, RNNCellFlex):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, learn_init_state=False, tensor_rank=DEFAULT_RANK,
               function= DEFAULT_FUNC, inner_function= DEFAULT_FUNC,
               activation=tanh, inner_activation=sigmoid,):
    super(GRUCellFlex, self).__init__(num_units, activation=activation)
    self._learn_init_state = learn_init_state
    self._tensor_rank = tensor_rank
    self._function = tf.make_template('output_function', function) #function
    self._inner_function = tf.make_template('inner_function', inner_function) #inner_function
    self._inner_activation = inner_activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def rank(self):
    return self._tensor_rank

  @property
  def learn_init_state(self):
    return self._learn_init_state

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with vs.variable_scope("Gates"):  # Reset gate and update gate.
        r, u = array_ops.split(_apply_func([inputs, state], self._tensor_rank,
                                             2 * self._num_units, self._inner_function), 2,self.rank-1 )
        r, u = sigmoid(r), sigmoid(u)
      with vs.variable_scope("Candidate"):
        # with tf.variable_scope("reset_portion"):
        #   reset_portion = r * _apply_func([state], self._tensor_rank,
        #                              self._num_units, self._function)
        # with tf.variable_scope("inputs_portion"):
        #   inputs_portion = _apply_func([inputs], self._tensor_rank,
        #                              self._num_units, self._function)
        # c = self._function(reset_portion + inputs_portion)
        c = self._activation(_apply_func([inputs, r * state], self._tensor_rank,
                                     self._num_units, self._function))
      new_h = u * state + (1 - u) * c

    return new_h, new_h


class MultiRNNCellFlex(MultiRNNCell,RNNCellFlex):
  """RNN cell composed sequentially of multiple simple cells."""

  @property
  def rank(self):
    return self._cells[0].rank

  def zero_state(self, prefix, dtype):
    new_states = []
    for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          new_states.append(cell.zero_state(prefix, dtype))

    zero_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat( new_states,self.rank-1))
    return zero_states

  def init_state(self, prefix, dtype):
    new_states = []
    for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          new_states.append(cell.init_state(prefix, dtype))

    init_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat( new_states,self.rank-1))
    return init_states

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_state_pos = 0
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          if self._state_is_tuple:
            if not nest.is_sequence(state):
              raise ValueError(
                  "Expected state to be a tuple of length %d, but received: %s"
                  % (len(self.state_size), state))
            cur_state = state[i]
          else:
            cur_state = array_ops.slice(state, [0] * (self.rank-1) + [cur_state_pos],
                                        [-1] * (self.rank-1) + [cell.state_size])
            cur_state_pos += cell.state_size
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat( new_states,self.rank-1))
    return cur_inp, new_states


  def init_call(self, inputs, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with vs.variable_scope("Cell%d" % i):
          cur_inp, new_state = cell.init_call(cur_inp)
          new_states.append(new_state)
    new_states = (tuple(new_states) if self._state_is_tuple
                  else array_ops.concat( new_states,self.rank-1))
    return cur_inp, new_states

class EmbeddingWrapperFlex(RNNCellFlex):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_func, embedding_size, normalizer=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    self._embedding_size = embedding_size
    self._cell = cell
    self._num_units = cell._num_units
    self._function = tf.make_template('embedding_function', embedding_func)
    if normalizer is not None:
      self._normalizer = tf.make_template('normalizer', normalizer, create_scope_now_=True)
    else:
      self._normalizer = None

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def rank(self):
    return self._cell.rank

  def zero_state(self, prefix, dtype):
    return self._cell.zero_state(prefix, dtype)

  def init_state(self, prefix, dtype):
    return self._cell.init_state(prefix, dtype)

  def init_call(self, inputs, scope=None):
    inputs = _apply_func(inputs, self.rank, self._embedding_size, self._function)
    return self._cell.init_call(inputs, scope=None)

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with vs.variable_scope(scope or type(self).__name__):
      embedded = _apply_func(inputs, self.rank, self._embedding_size, self._function)

      if self._normalizer is not None:
        embedded - self._normalizer(embedded)

    return self._cell(embedded, state)

class OutputProjectionWrapperFlex(RNNCellFlex):
  """Operator adding an output projection to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your outputs in time,
  do the projection on this batch-concatenated sequence, then split it
  if needed or directly feed into a softmax.
  """

  def __init__(self, cell,  num_units, function=DEFAULT_FUNC):
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if num_units < 1:
      raise ValueError("Parameter output_size must be > 0: %d." % num_units)
    self._cell = cell
    self._num_units = num_units
    self._function = tf.make_template('output_function', function)

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def rank(self):
    return self._cell.rank

  def zero_state(self, prefix, dtype):
    return self._cell.zero_state(prefix, dtype)

  def init_state(self, prefix, dtype):
    return self._cell.init_state(prefix, dtype)

  def init_call(self, inputs, scope=None):
    return self._cell.init_call(inputs, scope=None)

  def __call__(self, inputs, state, scope=None):
    """Run the cell and output projection on inputs, starting from state."""
    output, res_state = self._cell(inputs, state)
    # Default scope: "OutputProjectionWrapper"
    with vs.variable_scope(scope or type(self).__name__):
      projected = _apply_func(output, self.rank, self._num_units, self._function)
    return projected, res_state

def _apply_func(args, output_rank, output_size, func, scope=None):
    args = nest.flatten(args)
    incoming = array_ops.concat( args,output_rank-1)

    with vs.variable_scope(scope or "Apply_Func"):
        res = func(incoming,output_size)

    return res
