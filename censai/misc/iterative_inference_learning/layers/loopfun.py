import tensorflow as tf
from iterative_inference_learning.layers.array_ops import multi_slice

class LoopFunction(object):

  def __init__(self, input_function, output_function, stopping_function):
    self.input_function = input_function
    self.output_function = output_function
    self.stopping_function = stopping_function

  def __call__(self, time, cell, old_input, old_output, old_state, other_output):

    new_input = self.input_function(old_input, old_output , other_output)
    cell_output, new_state = cell(new_input, old_state)
    new_output = self.output_function(cell_output, old_output)
    next_finished = self.stopping_function(time, old_output, new_output)

    return next_finished, new_input, new_output, new_state


class GradientFunction(object):
  def __init__(self, error_fun, stop_grad=True):
    self.error_fun = error_fun
    self.stop_grad = stop_grad

  def __call__(self, x):
    loss = self.error_fun(x)
    grad = tf.gradients(loss, x)[0]
    if self.stop_grad:
      grad = tf.stop_gradient(grad)

    return grad

class ApplySplitFunction(object):
  def __init__(self, func, split_dim, num_splits):
    self.func = func
    self.split_dim = split_dim
    self.num_splits = num_splits

  def __call__(self, x , y):
    print('called ApplySplitFunction')
    print self.split_dim , self.num_splits , x , y
    #x_split = tf.split(self.split_dim, self.num_splits, x)
    x_split = tf.split(x, self.num_splits,self.split_dim)
    x_out = [self.func(x_ , y) for x_ in x_split]
    x_concat = tf.concat(x_out,self.split_dim)

    return x_concat

class ApplyMultFunction(object):
  def __init__(self, funcs, concat_dim):
    self.funcs = funcs
    self.concat_dim = concat_dim

  def __call__(self, x,y):
    x_out = [f_(x,y) for f_ in self.funcs]
    x_concat = tf.concat(x_out,self.concat_dim)

    return x_concat

class OutputFunction(object):
  def __init__(self, func, accum=True):
    self.func = func
    self.accum = accum

  def __call__(self, x_new, x_old=None):
    x_out = self.func(x_new)
    if self.accum:
      x_out = x_old + x_out

    return x_out

class StoppingFunction(object):
  def __init__(self, stop_time):
    self.stop_time = stop_time


  def __call__(self, time, *kwargs):
    return time >= self.stop_time

class InputFunction(object):
  def __init__(self, slice_dim, input_slices, input_funcs, output_slices, output_funcs):
    self.slice_dim = slice_dim
    self.input_slices = input_slices
    self.output_slices = output_slices
    self.input_funcs = [tf.make_template('input_fun_'+str(i), f) for i, f in enumerate(input_funcs)]
    self.output_funcs = [tf.make_template('output_fun_'+str(i), f) for i, f in enumerate(output_funcs)]

  def __call__(self, old_input, old_output, other_output):
    input_list = []
    sliced_input = multi_slice(self.slice_dim, self.input_slices, old_input)
    for x , f in zip(sliced_input, self.input_funcs):
      input_list.append(f(x))

    print('HERE !!!!')
    print(old_output.shape)
    print(other_output.shape)
    sliced_output = multi_slice(self.slice_dim, self.output_slices, old_output)
    sliced_other = multi_slice(self.slice_dim, self.output_slices, other_output)
    for x, y , f in zip(sliced_output, sliced_other , self.output_funcs):
      input_list.append(f(x,y))

    return tf.concat( input_list,self.slice_dim)

