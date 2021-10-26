import tensorflow as tf


# class SqueezeFlow(tf.keras.layers.Layer):
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, z, ldj, reverse=False):
#         B, H, W, C = z.shape
#         if reverse: # H/2 x W/2 x 4C ==> H x W x C
#         else: # H x W x C ==> H/2 x W/2 x 4C
