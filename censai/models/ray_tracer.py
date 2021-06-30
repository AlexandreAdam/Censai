import tensorflow as tf
from censai.models.layers import UnetDecodingLayer, UnetEncodingLayer
from censai.definitions import logkappa_normalization, log_kappa
from .utils import get_activation


class RayTracer(tf.keras.Model):
    def __init__(
            self,
            pixels,
            filter_scaling=1,
            layers=4,
            block_conv_layers=2,
            kernel_size=3,
            filters=32,
            strides=2,
            bottleneck_filters=None,
            resampling_kernel_size=None,
            upsampling_interpolation=False,     # use strided transposed convolution if false
            kernel_regularizer_amp=0.,
            bias_regularizer_amp=0.,            # if bias is used
            activation="linear",
            initializer="random_uniform",
            use_bias=False,
            kappalog=True,
            normalize=False,
            trainable=True,
            name="ray_tracer",
    ):
        super(RayTracer, self).__init__(name=name)
        self.trainable = trainable
        self.kappalog = kappalog
        self.kappa_normalize = normalize

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_regularizer_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L2(l2=bias_regularizer_amp)})

        resampling_kernel_size = resampling_kernel_size if resampling_kernel_size is not None else kernel_size
        bottleneck_filters = bottleneck_filters if bottleneck_filters is not None else int(filter_scaling**(layers) * filters)

        activation = get_activation(activation)

        # compute size of bottleneck here
        bottleneck_size = pixels // strides**(layers)

        self.encoding_layers = []
        self.decoding_layers = []
        for i in range(layers):
            self.encoding_layers.append(UnetEncodingLayer(
                kernel_size=kernel_size,
                downsampling_kernel_size=resampling_kernel_size,
                filters=int(filter_scaling**(i) * filters),
                strides=strides,
                conv_layers=block_conv_layers,
                activation=activation,
                **common_params
            ))
            self.decoding_layers.append(UnetDecodingLayer(
                kernel_size=kernel_size,
                upsampling_kernel_size=resampling_kernel_size,
                filters=int(filter_scaling**(i) * filters),
                conv_layers=block_conv_layers,
                strides=strides,
                activation=activation,
                bilinear=upsampling_interpolation,
                **common_params
            ))

        # reverse decoding layers order
        self.decoding_layers = self.decoding_layers[::-1]

        self.bottleneck_layer1 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=2*bottleneck_size,  # we perform a convolution over the full image at this point,
            activation="linear",
            **common_params
        )
        self.bottleneck_layer2 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=2*bottleneck_size,
            activation="linear",
            **common_params
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

    @tf.function
    def kappa_link(self, kappa):
        if self.kappalog:
            kappa = log_kappa(kappa)
            if self.kappa_normalize:
                return logkappa_normalization(kappa, forward=True)
            return kappa
        else:
            return kappa

    def __call__(self, kappa):
        return self.call(kappa)

    def call(self, kappa):
        kappa = self.kappa_link(kappa)  # preprocessing
        skip_connections = []
        z = kappa
        for i in range(len(self.encoding_layers)):
            c_i, z = self.encoding_layers[i](z)
            skip_connections.append(c_i)
        skip_connections = skip_connections[::-1]
        z = self.bottleneck_layer1(z)
        z = self.bottleneck_layer2(z)
        for i in range(len(self.decoding_layers)):
            z = self.decoding_layers[i](z, skip_connections[i])
        z = self.output_layer(z)
        return z

    def cost(self, kappa, alpha_true):
        alpha_pred = self.call(kappa)
        return tf.reduce_mean((alpha_pred - alpha_true) ** 2)
