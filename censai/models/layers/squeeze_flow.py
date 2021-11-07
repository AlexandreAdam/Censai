import tensorflow as tf


class SqueezeFlow(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, z, ldj, reverse=False):
        if reverse: # H/2 x W/2 x 4C ==> H x W x C
            z = tf.nn.depth_to_space(z, block_size=2)
        else: # H x W x C ==> H/2 x W/2 x 4C
            z = tf.nn.space_to_depth(z, block_size=2)
        return z, ldj
