import tensorflow as tf


def decode_all(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'kappa': tf.io.FixedLenFeature([], tf.string),
              'kappa pixels': tf.io.FixedLenFeature([], tf.int64),
              'Einstein radius before rescaling': tf.io.FixedLenFeature([], tf.float32),
              'Einstein radius': tf.io.FixedLenFeature([], tf.float32),
              'rescaling factor': tf.io.FixedLenFeature([], tf.float32),
              'z source': tf.io.FixedLenFeature([], tf.float32),
              'z lens': tf.io.FixedLenFeature([], tf.float32),
              'kappa fov': tf.io.FixedLenFeature([], tf.float32),
              'sigma crit': tf.io.FixedLenFeature([], tf.float32),
              'kappa id': tf.io.FixedLenFeature([], tf.int64)
          })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    kappa_pixels = example['kappa pixels']

    example['kappa'] = tf.reshape(kappa, [kappa_pixels, kappa_pixels, 1])
    return example


def decode_einstein_radii_info(record_bytes):
    params_keys = ['Einstein radius before rescaling', 'Einstein radius', 'rescaling factor']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]


def decode_shape(record_bytes):
    example = decode_all(record_bytes)
    return example["kappa pixels"]


def decode_train(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'kappa': tf.io.FixedLenFeature([], tf.string),
              'kappa pixels': tf.io.FixedLenFeature([], tf.int64),
          })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    kappa_pixels = example['kappa pixels']

    example['kappa'] = tf.reshape(kappa, [kappa_pixels, kappa_pixels, 1])
    return kappa

