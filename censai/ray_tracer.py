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
            kernel_regularizer_amp=1e-4,
            bias_regularizer_amp=1e-4,  # if bias is used
            activation="linear",
            batch_norm=False,
            use_bias=False,
            trainable=True,
            scaling=1,
            bottle_scaling=1,
            skip_connections=True,
            one_by_one_convs=True
    ):
        super(RayTracer512, self).__init__(name=name)
        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_regularizer_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L2(l2=bias_regularizer_amp)})
        main_kernel = tuple([decoder_encoder_kernel_size]*2)
        pre_bottle_kernel = tuple([pre_bottleneck_kernel_size]*2)
        filters = decoder_encoder_filters
        bottle_stride = tuple([bottleneck_strides]*2)
        self.trainable = trainable
        self.batch_norm = batch_norm
        self.skip_connections = skip_connections
        self.one_by_one_convs = one_by_one_convs
        if self.batch_norm:
            self.batch_norm_layers = []
            for i in range(11):
                self.batch_norm_layers.append(
                    tf.keras.layers.BatchNormalization()
                )
        if activation == "leaky_relu":
            activation = tf.keras.layers.LeakyReLU()
        elif activation == "gelu":
            activation = tf.keras.activations.gelu
        else:
            activation = tf.keras.layers.Activation(activation)

        self.Lc11 = tf.keras.layers.Conv2D(filters, main_kernel, activation=activation, **common_params)
        self.Lc12 = tf.keras.layers.Conv2D(filters, main_kernel, activation=activation, **common_params)
        self.Lp13 = tf.keras.layers.Conv2D(filters, main_kernel, activation=activation, strides=(2, 2), **common_params)  # 512 -> 256
        if self.one_by_one_convs:
            self.Lp14 = tf.keras.layers.Conv2D(filters, (1, 1), activation="linear", **common_params)

        self.Lc21 = tf.keras.layers.Conv2D(int(scaling*filters), main_kernel, activation=activation, **common_params)
        self.Lc22 = tf.keras.layers.Conv2D(int(scaling*filters), main_kernel, activation=activation, **common_params)
        self.Lp23 = tf.keras.layers.Conv2D(int(scaling*filters), main_kernel, activation=activation, strides=(2, 2), **common_params)  # 256 -> 128
        if self.one_by_on_convs:
            self.Lp24 = tf.keras.layers.Conv2D(int(scaling*filters), (1, 1), activation="linear", **common_params)

        self.Lc31 = tf.keras.layers.Conv2D(int(scaling**2*filters), main_kernel, activation=activation, **common_params)
        self.Lc32 = tf.keras.layers.Conv2D(int(scaling**2*filters), main_kernel, activation=activation, **common_params)
        self.Lp33 = tf.keras.layers.Conv2D(int(scaling**2*filters), main_kernel, activation=activation, strides=(2, 2), **common_params)  # 128 -> 64
        if self.one_by_one_convs:
            self.Lp34 = tf.keras.layers.Conv2D(int(scaling**2*filters), (1, 1), activation="linear", **common_params)

        self.Lc41 = tf.keras.layers.Conv2D(int(scaling**3*filters), main_kernel, activation=activation, **common_params)
        self.Lc42 = tf.keras.layers.Conv2D(int(scaling**3*filters), main_kernel, activation=activation, **common_params)
        self.Lp43 = tf.keras.layers.Conv2D(int(scaling**3*filters), main_kernel, activation=activation, strides=(2, 2), **common_params)  # 64 -> 32
        if self.one_by_one_convs:
            self.Lp44 = tf.keras.layers.Conv2D(int(scaling**3*filters), (1, 1), activation="linear", **common_params)

        self.Lc51 = tf.keras.layers.Conv2D(int(scaling**4*filters), main_kernel, activation=activation, **common_params)
        self.Lc52 = tf.keras.layers.Conv2D(int(scaling**4*filters), main_kernel, activation=activation, **common_params)
        self.Lp53 = tf.keras.layers.Conv2D(int(scaling**4*filters), pre_bottle_kernel, activation=activation, strides=bottle_stride, **common_params)  # 32 -> 8
        if self.one_by_one_convs:
            self.Lp54 = tf.keras.layers.Conv2D(int(scaling**4*filters), (1, 1), activation="linear", **common_params)

        self.LcZ1 = tf.keras.layers.Conv2D(int(bottle_scaling*filters), (16, 16), activation="linear", **common_params)  # Actual convolution at this stage (kernel size twice the image size)
        self.LcZ2 = tf.keras.layers.Conv2D(int(bottle_scaling*filters), (16, 16), activation="linear", **common_params)
        
        if self.one_by_one_convs:
            self.Lu60 = tf.keras.layers.Conv2D(int(scaling**4*filters), (1, 1), activation="linear", **common_params)
        if upsampling_interpolation:
            self.Lu61 = tf.keras.layers.UpSampling2D(size=bottle_stride, data_format=common_params["data_format"], interpolation="bilinear")
        else:
            self.Lu61 = tf.keras.layers.Conv2DTranspose(int(scaling**4*filters), pre_bottle_kernel, strides=bottleneck_strides, activation="linear", **common_params)  # 8 -> 32
        self.Lc62 = tf.keras.layers.Conv2D(int(scaling**4*filters), main_kernel, activation=activation, **common_params)
        self.Lc63 = tf.keras.layers.Conv2D(int(scaling**4*filters), main_kernel, activation=activation, **common_params)

        if self.one_by_one_convs:
            self.Lu70 = tf.keras.layers.Conv2D(int(scaling**3*filters), (1, 1), activation="linear", **common_params)
        if upsampling_interpolation:
            self.Lu71 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="bilinear")
        else:
            self.Lu71 = tf.keras.layers.Conv2DTranspose(int(scaling**3*filters), (2, 2), activation="linear", strides=(2, 2), **common_params)  # 32 -> 64
        self.Lc72 = tf.keras.layers.Conv2D(int(scaling**3*filters), main_kernel, activation=activation, **common_params)
        self.Lc73 = tf.keras.layers.Conv2D(int(scaling**3*filters), main_kernel, activation=activation, **common_params)

        if self.one_by_one_convs:
            self.Lu80 = tf.keras.layers.Conv2D(int(scaling**2*filters), (1, 1), activation="linear", **common_params)
        if upsampling_interpolation:
            self.Lu81 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="bilinear")
        else:
            self.Lu81 = tf.keras.layers.Conv2DTranspose(int(scaling**2*filters), (2, 2), activation="linear", strides=(2, 2), **common_params)  # 64 -> 128
        self.Lc82 = tf.keras.layers.Conv2D(int(scaling**2*filters), main_kernel, activation=activation, **common_params)
        self.Lc83 = tf.keras.layers.Conv2D(int(scaling**2*filters), main_kernel, activation=activation, **common_params)

        if self.one_by_one_convs:
            self.Lu90 = tf.keras.layers.Conv2D(int(scaling*filters), (1, 1), activation="linear", **common_params)
        if upsampling_interpolation:
            self.Lu91 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="bilinear")
        else:
            self.Lu91 = tf.keras.layers.Conv2DTranspose(int(scaling*filters), (2, 2), activation="linear", strides=(2, 2), **common_params)  # 128 -> 256
        self.Lc92 = tf.keras.layers.Conv2D(int(scaling*filters), main_kernel, activation=activation, **common_params)
        self.Lc93 = tf.keras.layers.Conv2D(int(scaling*filters), main_kernel, activation=activation, **common_params)
        
        if self.one_by_one_convs:
            self.Lu100 = tf.keras.layers.Conv2D(filters, (1, 1), activation="linear", **common_params)
        if upsampling_interpolation:
            self.Lu101 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format=common_params["data_format"], interpolation="bilinear")
        else:
            self.Lu101 = tf.keras.layers.Conv2DTranspose(filters, (2, 2), activation="linear", strides=(2, 2), **common_params)  # 256 -> 512
        self.Lc102 = tf.keras.layers.Conv2D(filters, main_kernel, activation=activation, **common_params)
        self.Lc103 = tf.keras.layers.Conv2D(filters, main_kernel, activation=activation, **common_params)

        self.Loutputs = tf.keras.layers.Conv2D(2, (1, 1), activation="linear", **common_params) # rescaling of ouptut

    def call(self, kappa):
        if self.batch_norm:
            c1 = self.batch_norm_layers[0](kappa)
        else:
            c1 = kappa
        c1 = self.Lc11(c1)
        c1 = self.Lc12(c1)  # keep this for skip connection
        c2 = self.Lp13(c1)  # downsample
        if self.one_by_one_convs:
            c2 = self.Lp14(c2) # rescale with a 1by1 conv

        if self.batch_norm:
            c2 = self.batch_norm_layers[1](c2)
        c2 = self.Lc21(c2)
        c2 = self.Lc22(c2)
        c3 = self.Lp23(c2)
        if self.one_by_one_convs:
            c3 = self.Lp24(c3)

        if self.batch_norm:
            c3 = self.batch_norm_layers[2](c3)
        c3 = self.Lc31(c3)
        c3 = self.Lc32(c3)
        c4 = self.Lp33(c3)
        if self.one_by_one_convs:
            c4 = self.Lp34(c4)

        if self.batch_norm:
            c4 = self.batch_norm_layers[3](c4)
        c4 = self.Lc41(c4)
        c4 = self.Lc42(c4)
        c5 = self.Lp43(c4)
        if self.one_by_one_convs:
            c5 = self.Lp44(c5)

        if self.batch_norm:
            c5 = self.batch_norm_layers[4](c5)
        c5 = self.Lc51(c5)
        c5 = self.Lc52(c5)
        z = self.Lp53(c5)  # from here on, we use a single variable z to reduce memory consumption
        if self.one_by_one_convs:
            z = self.Lp54(z)

        if self.batch_norm:  # should we put a batch norm here?
            z = self.batch_norm_layers[5](z)
        z = self.LcZ1(z)
        z = self.LcZ2(z)

        if self.one_by_one_convs:
            z = self.Lu60(z)
        z = self.Lu61(z)  # upsampling
        if self.skip_connections:
            z = tf.concat([z, c5], axis=3)  # skip connection
        if self.batch_norm:
            z = self.batch_norm_layers[6](z)
        z = self.Lc62(z)
        z = self.Lc63(z)
        
        if self.one_by_one_convs:
            z = self.Lu70(z)
        z = self.Lu71(z)
        if self.skip_connections:
            z = tf.concat([z, c4], axis=3)
        if self.batch_norm:
            z = self.batch_norm_layers[7](z)
        z = self.Lc72(z)
        z = self.Lc73(z)
        
        if self.one_by_one_convs:
            z = self.Lu80(z) 
        z = self.Lu81(z)
        if self.skip_connections:
            z = tf.concat([z, c3], axis=3)
        if self.batch_norm:
            z = self.batch_norm_layers[8](z)
        z = self.Lc82(z)
        z = self.Lc83(z)
        
        if self.one_by_one_convs:
            z = self.Lu90(z)
        z = self.Lu91(z)
        if self.skip_connections:
            z = tf.concat([z, c2], axis=3)
        if self.batch_norm:
            z = self.batch_norm_layers[9](z)
        z = self.Lc92(z)
        z = self.Lc93(z)
        
        if self.one_by_one_convs:
            z = self.Lu100(z)
        z = self.Lu101(z)
        if self.skip_connections:
            z = tf.concat([z, c1], axis=3)
        if self.batch_norm:
            z = self.batch_norm_layers[10](z)
        z = self.Lc102(z)
        z = self.Lc103(z)

        z = self.Loutputs(z)

        return z

    def cost(self, kappa, alpha_true):
        alpha_pred = self.call(kappa)
        # alpha_label = tf.concat([x_a_label, y_a_label] , axis=3)
        return tf.reduce_mean((alpha_pred - alpha_true)**2)
