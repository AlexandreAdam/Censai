import tensorflow as tf
from censai.definitions import DTYPE


def checkerboard_mask(pixels, invert=False):
    x = tf.range(pixels, dtype=DTYPE)
    x, y = tf.meshgrid(x, x)
    mask = tf.math.floormod(x + y, 2)[tf.newaxis, tf.newaxis, ...]
    if invert:
        return 1 - mask
    else:
        return mask


def channel_mask(c_in, invert=False):
    mask = tf.concat([tf.ones(c_in//2, dtype=DTYPE), tf.zeros(c_in - c_in//2, dtype=DTYPE)])
    mask = mask[tf.newaxis, tf.newaxis, tf.newaxis, :]
    if invert:
        return 1 - mask
    else:
        return mask