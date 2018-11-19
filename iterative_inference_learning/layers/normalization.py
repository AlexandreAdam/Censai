from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import moving_averages
import tensorflow as tf

@add_arg_scope
def layer_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  Can be used as a normalizer function for conv2d and fully_connected.

  Args:
    inputs: a tensor with 2 or more dimensions. The normalization
            occurs over all but the first dimension.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if inputs_rank > 2:
        axis = list(range(1, inputs_rank-1)) #list(range(1, inputs_rank))
    else:
        axis = list(range(1, inputs_rank))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    mean, variance = nn.moments(inputs, axis, keep_dims=True)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope,
                                       outputs)

@add_arg_scope
def global_norm(inputs,
               center=True,
               scale=True,
               activation_fn=None,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               scope=None):
  """Adds a Layer Normalization layer from https://arxiv.org/abs/1607.06450.

    "Layer Normalization"

    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

  Can be used as a normalizer function for conv2d and fully_connected.

  Args:
    inputs: a tensor with 2 or more dimensions. The normalization
            occurs over all but the first dimension.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_op_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """
  with variable_scope.variable_scope(scope, 'LayerNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    if inputs_rank > 2:
        axis = list(range(1, inputs_rank-1)) #list(range(1, inputs_rank))
    else:
        axis = list(range(1, inputs_rank))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None
    if center:
      beta_collections = utils.get_variable_collections(variables_collections,
                                                        'beta')
      beta = variables.model_variable('beta',
                                      shape=params_shape,
                                      dtype=dtype,
                                      initializer=init_ops.zeros_initializer,
                                      collections=beta_collections,
                                      trainable=trainable)
    if scale:
      gamma_collections = utils.get_variable_collections(variables_collections,
                                                         'gamma')
      gamma = variables.model_variable('gamma',
                                       shape=params_shape,
                                       dtype=dtype,
                                       initializer=init_ops.ones_initializer,
                                       collections=gamma_collections,
                                       trainable=trainable)
    # Calculate the moments on the last axis (layer activations).
    #mean, variance = nn.moments(inputs, axis, keep_dims=True)

    mean, variance = tf.zeros_like(inputs), tf.ones_like(inputs)
    # Compute layer normalization using the batch_normalization function.
    variance_epsilon = 1E-12
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, variance_epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope,
                                       outputs)


@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               initializers={},
               updates_collections=None,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=False,
               scope=None):
  """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

    "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift"

    Sergey Ioffe, Christian Szegedy

  Can be used as a normalizer function for conv2d and fully_connected.

  Note: When is_training is True the moving_mean and moving_variance need to be
  updated, by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so
  they need to be added as a dependency to the train_op, example:

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops)
      total_loss = control_flow_ops.with_dependencies([updates], total_loss)

  One can set update_collections=None to force the updates in place, but that
  can have speed penalty, specially in distributed settings.

  Args:
    inputs: a tensor with 2 or more dimensions, where the first dimension has
      `batch_size`. The normalization is over all but the last dimension.
    decay: decay for the moving average.
    center: If True, subtract `beta`. If False, `beta` is ignored.
    scale: If True, multiply by `gamma`. If False, `gamma` is
      not used. When the next layer is linear (also e.g. `nn.relu`), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    activation_fn: activation function, default set to None to skip it and
      maintain a linear activation.
    updates_collections: collections to collect the update ops for computation.
      The updates_ops need to be excuted with the train_op.
      If None, a control dependency would be added to make sure the updates are
      computed in place.
    is_training: whether or not the layer is in training mode. In training mode
      it would accumulate the statistics of the moments into `moving_mean` and
      `moving_variance` using an exponential moving average with the given
      `decay`. When it is not in training mode then it would use the values of
      the `moving_mean` and the `moving_variance`.
    reuse: whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: optional collections for the variables.
    outputs_collections: collections to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A `Tensor` representing the output of the operation.

  Raises:
    ValueError: if rank or last dimension of `inputs` is undefined.
  """

  with variable_scope.variable_scope(scope, 'BatchNorm', [inputs],
                                     reuse=reuse) as sc:
    inputs = ops.convert_to_tensor(inputs)
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims
    if inputs_rank is None:
      raise ValueError('Inputs %s has undefined rank.' % inputs.name)
    dtype = inputs.dtype.base_dtype
    axis = list(range(inputs_rank - 1))
    params_shape = inputs_shape[-1:]
    if not params_shape.is_fully_defined():
      raise ValueError('Inputs %s has undefined last dimension %s.' % (
          inputs.name, params_shape))
    # Allocate parameters for the beta and gamma of the normalization.
    beta, gamma = None, None

    # Create moving_mean and moving_variance variables and add them to the
    # appropiate collections.
    moving_mean_initializer = initializers.get('moving_mean', init_ops.zeros_initializer)
    moving_mean = variables.model_variable(
        'moving_mean',
        shape=params_shape,
        dtype=dtype,
        initializer=moving_mean_initializer,
        trainable=False)
    moving_variance_initializer = initializers.get('moving_variance', init_ops.ones_initializer)
    moving_variance = variables.model_variable(
        'moving_variance',
        shape=params_shape,
        dtype=dtype,
        initializer=moving_variance_initializer,
        trainable=False)

    # If `is_training` doesn't have a constant value, because it is a `Tensor`,
    # a `Variable` or `Placeholder` then is_training_value will be None and
    # `needs_moments` will be true.
    is_training_value = utils.constant_value(is_training)
    need_moments = is_training_value is None or is_training_value
    if need_moments:
      # Calculate the moments based on the individual batch.
      # Use a copy of moving_mean as a shift to compute more reliable moments.
      shift = math_ops.add(moving_mean, 0)
      mean, variance = nn.moments(inputs, axis, shift=shift)
      moving_vars_fn = lambda: (moving_mean, moving_variance)
      if updates_collections is None:
        def _force_updates():
          """Internal function forces updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          with ops.control_dependencies([update_moving_mean,
                                         update_moving_variance]):
            return array_ops.identity(mean), array_ops.identity(variance)
        mean, variance = utils.smart_cond(is_training,
                                          _force_updates,
                                          moving_vars_fn)
      else:
        def _delay_updates():
          """Internal function that delay updates moving_vars if is_training."""
          update_moving_mean = moving_averages.assign_moving_average(
              moving_mean, mean, decay)
          update_moving_variance = moving_averages.assign_moving_average(
              moving_variance, variance, decay)
          return update_moving_mean, update_moving_variance

        update_mean, update_variance = utils.smart_cond(is_training,
                                                        _delay_updates,
                                                        moving_vars_fn)
        ops.add_to_collections(updates_collections, update_mean)
        ops.add_to_collections(updates_collections, update_variance)
        # Use computed moments during training and moving_vars otherwise.
        vars_fn = lambda: (mean, variance)
        mean, variance = utils.smart_cond(is_training, vars_fn, moving_vars_fn)
    else:
      mean, variance = moving_mean, moving_variance
    # Compute batch_normalization.
    outputs = nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs_shape)
    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return utils.collect_named_outputs(outputs_collections,
                                       sc.original_name_scope, outputs)
