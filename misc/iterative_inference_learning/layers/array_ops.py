import tensorflow as tf

def multi_slice(split_dim, slices, value, name='multi_slices'):
  value_array = tf.unstack(value, axis=split_dim)
  output_list = []
  for s in slices:
    output_list.append(tf.stack(value_array[s[0]:s[0]+s[1]],split_dim))

  return output_list