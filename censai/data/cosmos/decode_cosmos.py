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


def decode(record_bytes):
    keys = ['image', 'psf', 'ps']
    example = decode_all(record_bytes)
    return [example[key] for key in keys]


def preprocess(image, psf, ps):
    image = tf.nn.relu(image)  # set negative pixels to 0.
    image = image / tf.reduce_max(image)  # set peak value to 1
    return image, psf, ps


def decode_image(record_bytes):
    example = decode_all(record_bytes)
    return example['image']


def preprocess_image(image):
    image = tf.nn.relu(image)  # set negative pixels to 0.
    image = image / tf.reduce_max(image)  # set peak value to 1
    return image


def decode_image_shape(record_bytes):
    keys = ['height']
    example = decode_all(record_bytes)
    return [example[key] for key in keys]




# # some tests that everything works
if __name__ == '__main__':
    import os
    path = os.path.join("/home/alexandre/Desktop/Projects", "Censai/data/cosmos_record_1.tfrecords")
    data = tf.data.TFRecordDataset(path)
    data = data.map(decode)
    data = data.batch(10)
    for (im, psf, ps) in data.as_numpy_iterator():
        break