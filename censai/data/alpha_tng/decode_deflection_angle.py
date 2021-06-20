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
              'Einstein radius': tf.io.FixedLenFeature([], tf.float32)
          })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    alpha = tf.io.decode_raw(example['alpha'], tf.float32)
    theta_e = example['Einstein radius']
    pixels = example['pixels']
    kappa_id = example['kappa_id']

    kappa = tf.reshape(kappa, [pixels, pixels, 1])
    alpha = tf.reshape(alpha, [pixels, pixels, 2])
    return kappa, alpha, theta_e, kappa_id


def decode_train(record_bytes):
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
            'Einstein radius': tf.io.FixedLenFeature([], tf.float32)
        })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    alpha = tf.io.decode_raw(example['alpha'], tf.float32)
    pixels = example['pixels']

    kappa = tf.reshape(kappa, [pixels, pixels, 1])
    alpha = tf.reshape(alpha, [pixels, pixels, 2])
    return kappa, alpha
