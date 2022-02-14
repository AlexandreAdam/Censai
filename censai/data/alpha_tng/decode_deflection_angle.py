import tensorflow as tf


def decode_all(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'kappa': tf.io.FixedLenFeature([], tf.string),
              'pixels': tf.io.FixedLenFeature([], tf.int64),
              'alpha': tf.io.FixedLenFeature([], tf.string),
              'rescale': tf.io.FixedLenFeature([], tf.float32),
              'kappa_id': tf.io.FixedLenFeature([], tf.int64),
              'Einstein radius': tf.io.FixedLenFeature([], tf.float32),
              'image_fov': tf.io.FixedLenFeature([], tf.float32),
              'kappa_fov': tf.io.FixedLenFeature([], tf.float32)
          })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    alpha = tf.io.decode_raw(example['alpha'], tf.float32)
    pixels = example['pixels']

    example['kappa'] = tf.reshape(kappa, [pixels, pixels, 1])
    example['alpha'] = tf.reshape(alpha, [pixels, pixels, 2])
    return example


def decode_train(record_bytes):
    params_keys = ['kappa', 'alpha']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]


def decode_physical_info(record_bytes):
    params_keys = ['image_fov', 'kappa_fov']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]
