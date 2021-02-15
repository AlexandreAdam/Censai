import tensorflow as tf
from tensorflow.python.util import nest
from iterative_inference_learning.layers.array_ops import multi_slice

class NameWrapper(object):
  def __init__(self, name_dict, split_dim):
    self.name_dict = name_dict
    self.split_dim = split_dim
    self.name_dict.update({'all': [0, self.num_dim]})

  @staticmethod
  def output_specs_toname(output_specs, split_dim):
    temp_dict = {}
    running_idx = 0

    for name, val in output_specs.items():
        temp_dict.update({name:[running_idx,val]})
        running_idx = running_idx + val

    return NameWrapper(temp_dict, split_dim)

  @property
  def num_dim(self):
    return max([sum(v) for v in self.name_dict.values()])

  def __call__(self, tensor, names, split_dim=None):
    if split_dim is None:
      split_dim = self.split_dim
    if nest.is_sequence(names):
      return multi_slice(split_dim, [self.name_dict[n] for n in names], tensor)
    else:
      return multi_slice(split_dim, [self.name_dict[names]], tensor)[0]


  def insert(self, tensor, name, new_input, split_dim=None):
    if split_dim is None:
      split_dim = self.split_dim
    rank = split_dim + 1
    idx = self.name_dict[name]
    top_slice = tf.slice(tensor, rank * [0], (rank - 1) * [-1] + [idx[0]])
    bottom_slice = tf.slice(tensor, (rank - 1) * [0] + [idx[0] + idx[1]], rank * [-1])
    output_tensor = tf.concat( [top_slice, new_input, bottom_slice],split_dim)

    return output_tensor
