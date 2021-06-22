import tensorflow as tf


def decode_all(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'kappa': tf.io.FixedLenFeature([], tf.string),
              'source': tf.io.FixedLenFeature([], tf.string),
              'lens': tf.io.FixedLenFeature([], tf.string),
              'Einstein radius before rescaling': tf.io.FixedLenFeature([], tf.float32),
              'Einstein radius': tf.io.FixedLenFeature([], tf.float32),
              'rescaling factor': tf.io.FixedLenFeature([], tf.float32),
              'power spectrum': tf.io.FixedLenFeature([], tf.string),
              'z source': tf.io.FixedLenFeature([], tf.float32),
              'z lens': tf.io.FixedLenFeature([], tf.float32),
              'image fov': tf.io.FixedLenFeature([], tf.float32),
              'kappa fov': tf.io.FixedLenFeature([], tf.float32),
              'sigma crit': tf.io.FixedLenFeature([], tf.float32),
              'src pixels': tf.io.FixedLenFeature([], tf.int64),
              'kappa pixels': tf.io.FixedLenFeature([], tf.int64),
              'noise rms': tf.io.FixedLenFeature([], tf.float32),
              "psf sigma": tf.io.FixedLenFeature([], tf.float32),
              'kappa id': tf.io.FixedLenFeature([], tf.int64)
          })
    kappa = tf.io.decode_raw(example['kappa'], tf.float32)
    source = tf.io.decode_raw(example['source'], tf.float32)
    lens = tf.io.decode_raw(example['lens'], tf.float32)
    ps = tf.io.decode_raw(example['power spectrum'], tf.float32)
    kappa_pixels = example['kappa pixels']
    source_pixels = example['src pixels']

    example['kappa'] = tf.reshape(kappa, [kappa_pixels, kappa_pixels, 1])
    example['source'] = tf.reshape(source, [source_pixels, source_pixels, 1])
    example['power spectrum'] = tf.reshape(ps, [source_pixels, source_pixels//2 + 1, 1])
    example['lens'] = tf.reshape(lens, [kappa_pixels, kappa_pixels, 1])
    return example


def decode_physical_model_info(record_bytes):
    params_keys = ['image fov', 'kappa fov', 'src pixels', 'kappa pixels', 'noise rms', 'psf sigma']
    example = decode_all(record_bytes)
    return {key: example[key] for key in params_keys}


def decode_einstein_radii_info(record_bytes):
    params_keys = ['Einstein radius before rescaling', 'Einstein radius', 'rescaling factor']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]


def decode_train(record_bytes):
    params_keys = ['lens', 'source', 'kappa']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]


def decode_train_with_ps(record_bytes):
    params_keys = ['lens', 'source', 'ps', 'kappa']
    example = decode_all(record_bytes)
    return [example[key] for key in params_keys]
