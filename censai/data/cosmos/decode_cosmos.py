import tensorflow as tf


def decode_all(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'image': tf.io.FixedLenFeature([], tf.string),
              'height': tf.io.FixedLenFeature([], tf.int64),
              'width': tf.io.FixedLenFeature([], tf.int64),
              'psf': tf.io.FixedLenFeature([], tf.string),
              'ps': tf.io.FixedLenFeature([], tf.string)
          })
    #  decode raw data to float tensors, assuming everything was encoded as float32
    image = tf.io.decode_raw(example["image"], tf.float32)
    psf = tf.io.decode_raw(example["psf"], tf.float32)
    ps = tf.io.decode_raw(example["ps"], tf.float32)
    h = example["height"]
    w = example["width"]
    image = tf.reshape(image, [h, w, 1])
    psf = tf.reshape(psf, [2*h, w + 1, 1])  # because of noise padding
    ps = tf.reshape(ps, [h, w//2 + 1, 1])
    example['image'] = image
    example['psf'] = psf
    example['ps'] = ps
    return example


def decode_image(record_bytes):
    example = tf.io.parse_single_example(
          # Data
          record_bytes,
          # Schema
          features={
              'image': tf.io.FixedLenFeature([], tf.string),
              'height': tf.io.FixedLenFeature([], tf.int64),
          })
    #  decode raw data to float tensors, assuming everything was encoded as float32
    image = tf.io.decode_raw(example["image"], tf.float32)
    h = example["height"]
    image = tf.reshape(image, [h, h, 1])
    example['image'] = image
    return example


def decode(record_bytes):
    keys = ['image', 'psf', 'ps']
    example = decode_all(record_bytes)
    return [example[key] for key in keys]


def preprocess(image, psf, ps):
    image = tf.nn.relu(image)  # set negative pixels to 0.
    image = image / tf.reduce_max(image)  # set peak value to 1
    return image, psf, ps


def preprocess_image(image):
    image = tf.nn.relu(image)  # set negative pixels to 0.
    image = image / tf.reduce_max(image)  # set peak value to 1
    return image


def decode_shape(record_bytes):
    example = decode_image(record_bytes)
    return example['height']
