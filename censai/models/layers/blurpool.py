import tensorflow as tf
import numpy as np


class BlurPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.kernel_size == 2:
            bk = np.array([[1, 1],
                           [1, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 3:
            bk = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]])
            bk = bk / np.sum(bk)
        elif self.kernel_size == 5:
            bk = np.array([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]])
            bk = bk / np.sum(bk)
        else:
            raise ValueError

        bk = np.repeat(bk, input_shape[3])

        bk = np.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        blur_init = tf.keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, self.kernel_size, input_shape[3], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool2D, self).build(input_shape)

    def call(self, x):
        x = tf.nn.depthwise_conv2d(x, self.blur_kernel, padding='same', strides=(self.pool_size, self.pool_size))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]