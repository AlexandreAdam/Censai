import tensorflow as tf
from censai.definitions import DTYPE


class ConvBlock(tf.keras.layers.Layer):
    """
    Conv block from ResNet architecture
    """
    def __init__(self, filters, kernel_size, kernel_reg_amp, bias_reg_amp, alpha=0.3, **kwargs):
        super(ConvBlock, self).__init__(dtype=DTYPE, **kwargs)
        self.non_linearity = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.batch_norm1 = tf.keras.layers.BatchNormalization(scale=False)
        self.batch_norm2 = tf.keras.layers.BatchNormalization(scale=False)
        self.batch_norm3 = tf.keras.layers.BatchNormalization(scale=False)

        self.conv_rescale1 = tf.keras.layers.Conv2D(
            filters=filters//2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.RandomUniform()
        )
        self.conv_rescale2 = tf.keras.layers.Conv2D(
            filters=filters//2,
            kernel_size=1,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.RandomUniform()
        )

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
            bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
        )

    def call(self, x, training=True):
        x = self.batch_norm1(x, training=training)
        x = self.non_linearity(x)
        x = self.conv_rescale1(x, training=training)
        x = self.batch_norm2(x, training=training)
        x = self.non_linearity(x)
        x = self.conv(x, training=training)
        x = self.batch_norm3(x, training=training)
        x = self.non_linearity(x)
        x = self.conv_rescale2(x, training=training)
        return x
