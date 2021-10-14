import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization


class UpsamplingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            filters,
            kernel_size,
            strides,
            group_norm,
            groups=1,
            **kwargs
    ):
        super(UpsamplingLayer, self).__init__()
        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            **kwargs
        )
        self.batch_norm = GroupNormalization(groups) if group_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x))

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return x


class PSP(tf.keras.layers.Layer):

    def __init__(self, filters, pixels, scaling:int=2, bilinear=True, group_norm=True, groups=1):
        super(PSP, self).__init__()
        self.max_pool1 = tf.keras.layers.MaxPool2D(pool_size=pixels)
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=pixels//scaling)
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=pixels//scaling**2)
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=pixels//scaling**3)

        self.conv1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding="SAME")
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding="SAME")
        self.conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding="SAME")
        self.conv4 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, padding="SAME")

        if bilinear:
            self.upsample1 = tf.keras.layers.UpSampling2D(size=pixels, interpolation="bilinear")
            self.upsample2 = tf.keras.layers.UpSampling2D(size=pixels//scaling, interpolation="bilinear")
            self.upsample3 = tf.keras.layers.UpSampling2D(size=pixels//scaling**2, interpolation="bilinear")
            self.upsample4 = tf.keras.layers.UpSampling2D(size=pixels//scaling**3, interpolation="bilinear")
        else:
            self.upsample1 = UpsamplingLayer(filters=1, kernel_size=1, strides=pixels, batch_norm=group_norm, groups=groups, padding="SAME")
            self.upsample2 = UpsamplingLayer(filters=1, kernel_size=1, strides=pixels//scaling, batch_norm=group_norm, groups=groups, padding="SAME")
            self.upsample3 = UpsamplingLayer(filters=1, kernel_size=1, strides=pixels//scaling**2, batch_norm=group_norm, groups=groups, padding="SAME")
            self.upsample4 = UpsamplingLayer(filters=1, kernel_size=1, strides=pixels//scaling**3, batch_norm=group_norm, groups=groups, padding="SAME")

        self.conv_out = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, padding="SAME")
        self.batch_norm_out = tf.keras.layers.BatchNormalization() if group_norm else tf.identity

    def call(self, x):
        x1, x2, x3, x4 = tf.split(tf.identity(x), 4, axis=3)
        x1 = self.max_pool1(x1)
        x2 = self.max_pool2(x2)
        x3 = self.max_pool3(x3)
        x4 = self.max_pool4(x4)
        
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        
        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        x4 = self.upsample4(x4)
        out = tf.concat([x, x1, x2, x3, x4], axis=3)
        out = self.conv_out(out)
        out = self.batch_norm_out(out)
        return out