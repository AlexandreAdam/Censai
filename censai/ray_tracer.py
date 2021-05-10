import tensorflow as tf


class RayTracer512(tf.keras.Model):
    def __init__(
            self,
            name="ray_tracer_512",
            initializer="random_uniform",
            bottleneck_kernel_size=16,
            bottleneck_strides=4,
            pre_bottleneck_kernel_size=6,
            decoder_encoder_kernel_size=3,
            decoder_encoder_filters=32,
            upsampling_interpolation=False,  # use strided transposed convolution if false
            trainable=True,
    ):
        super(RayTracer512, self).__init__(name=name)
        common_params = {"activation": "linear", "padding": "same", "kernel_initializer": initializer, "data_format": "channels_last"}
        main_kernel = tuple([decoder_encoder_kernel_size]*2)
        pre_bottle_kernel = tuple([pre_bottleneck_kernel_size]*2)
        filters = decoder_encoder_filters
        bottle_stride = tuple([bottleneck_strides]*2)
        self.trainable = trainable

        self.Lc11 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc12 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lp13 = tf.keras.layers.Conv2D(filters, main_kernel, strides=(2, 2), **common_params)  # 512 -> 256

        self.Lc21 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc22 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lp23 = tf.keras.layers.Conv2D(filters, main_kernel, strides=(2, 2), **common_params)  # 256 -> 128

        self.Lc31 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lcfilters = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lp33 = tf.keras.layers.Conv2D(filters, main_kernel, strides=(2, 2), **common_params)  # 128 -> 64

        self.Lc41 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc42 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lp43 = tf.keras.layers.Conv2D(filters, main_kernel, strides=(2, 2), **common_params)  # 64 -> 32

        self.Lc51 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc52 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lp53 = tf.keras.layers.Conv2D(filters, pre_bottle_kernel, strides=bottle_stride, **common_params)  # 32 -> 8

        self.LcZ1 = tf.keras.layers.Conv2D(filters, (16, 16), **common_params)  # Actual convolution at this stage (kernel size twice the image size)
        self.LcZ2 = tf.keras.layers.Conv2D(filters, (16, 16), **common_params)

        if upsampling_interpolation:
            self.Lu61 = tf.keras.layers.UpSampling2D(size=bottle_stride, data_format=common_params["data_format"], interpolation="linear")
        else:
            self.Lu61 = tf.keras.layers.Conv2DTranspose(filters, pre_bottle_kernel, strides=bottleneck_strides, **common_params)  # 8 -> 32
        self.Lc62 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc63 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)

        if upsampling_interpolation:
            self.Lu71 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="linear")
        else:
            self.Lu71 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), **common_params)  # 32 -> 64
        self.Lc72 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc73 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)

        if upsampling_interpolation:
            self.Lu81 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="linear")
        else:
            self.Lu81 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), **common_params)  # 64 -> 128
        self.Lc82 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc83 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)

        if upsampling_interpolation:
            self.Lu91 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="linear")
        else:
            self.Lu91 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), **common_params)  # 128 -> 256
        self.Lc92 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc93 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)

        if upsampling_interpolation:
            self.Lu101 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="linear")
        else:
            self.Lu101 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), **common_params)  # 256 -> 512
        self.Lc102 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)
        self.Lc103 = tf.keras.layers.Conv2D(filters, main_kernel, **common_params)

        self.Loutputs = tf.keras.layers.Conv2D(2, main_kernel, **common_params)

    def call(self, kappa):
        c1 = self.Lc11(kappa)
        c1 = self.Lc12(c1)  # keep this for skip connection
        c2 = self.Lp13(c1)  # downsample

        c2 = self.Lc21(c2)
        c2 = self.Lc22(c2)
        c3 = self.Lp23(c2)

        c3 = self.Lc31(c3)
        c3 = self.Lcfilters(c3)
        c4 = self.Lp33(c3)

        c4 = self.Lc41(c4)
        c4 = self.Lc42(c4)
        c5 = self.Lp43(c4)

        c5 = self.Lc51(c5)
        c5 = self.Lc52(c5)
        z = self.Lp53(c5)  # from here on, we use a single variable z to reduce memory consumption

        z = self.LcZ1(z)
        z = self.LcZ2(z)

        z = self.Lu61(z)  # upsampling
        z = tf.concat([z, c5], axis=3)  # skip connection
        z = self.Lc62(z)
        z = self.Lc63(z)

        z = self.Lu71(z)
        z = tf.concat([z, c4], axis=3)
        z = self.Lc72(z)
        z = self.Lc73(z)

        z = self.Lu81(z)
        z = tf.concat([z, c3], axis=3)
        z = self.Lc82(z)
        z = self.Lc83(z)

        z = self.Lu91(z)
        z = tf.concat([z, c2], axis=3)
        z = self.Lc92(z)
        z = self.Lc93(z)

        z = self.Lu101(z)
        z = tf.concat([z, c1], axis=3)
        z = self.Lc102(z)
        z = self.Lc103(z)

        z = self.Loutputs(z)

        return z

    def cost(self, kappa, alpha_true):
        alpha_pred = self.call(kappa)
        # alpha_label = tf.concat([x_a_label, y_a_label] , axis=3)
        return tf.reduce_mean((alpha_pred - alpha_true)**2)#, alpha