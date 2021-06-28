import tensorflow as tf
from censai.definitions import DTYPE


def decode(record_bytes):
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
    return image, psf, ps


def preprocess(images, psf, ps):
    images = tf.where(images < 0, tf.constant(0, DTYPE), images)  # set negative pixel to 0
    return images, psf, ps


# # some tests that everything works
if __name__ == '__main__':
    import os
    path = os.path.join("/home/alexandre/Desktop/Projects", "Censai/data/cosmos_record_1.tfrecords")
    data = tf.data.TFRecordDataset(path)
    data = data.map(decode)
    data = data.batch(10)
    for (im, psf, ps) in data.as_numpy_iterator():
        break