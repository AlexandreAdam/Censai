import tensorflow as tf
from censai.models.layers import UnetDecodingLayer, UnetEncodingLayer
from .utils import get_activation


class RayTracer(tf.keras.Model):
    def __init__(
            self,
            pixels,
            name="ray_tracer",
            initializer="random_uniform",
            bottleneck_kernel_size=16,
            bottleneck_strides=4,
            bottleneck_filters=64,
            pre_bottleneck_kernel_size=6,
            decoder_encoder_kernel_size=3,
            decoder_encoder_filters=32,
            layers=4,
            block_conv_layers=2,
            upsampling_interpolation=False,  # use strided transposed convolution if false
            kernel_regularizer_amp=0.,
            bias_regularizer_amp=0.,  # if bias is used
            activation="linear",
            use_bias=False,
            trainable=True,
            filter_scaling=1,
    ):
        super(RayTracer, self).__init__(name=name)
        self.trainable = trainable

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_regularizer_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L2(l2=bias_regularizer_amp)})
        main_kernel = tuple([decoder_encoder_kernel_size] * 2)
        pre_bottle_kernel = tuple([pre_bottleneck_kernel_size] * 2)
        bottle_kernel = tuple([bottleneck_kernel_size] * 2)
        filters = decoder_encoder_filters
        bottle_stride = tuple([bottleneck_strides] * 2)
        activation = get_activation(activation)
        # compute size of bottleneck here
        bottleneck_size = pixels / (layers -1) / 2 / bottleneck_strides

        self.encoding_layers = []
        self.decoding_layers = []
        for i in range(layers-1):
            self.encoding_layers.append(UnetEncodingLayer(
                kernel_size=main_kernel,
                filters=int(filter_scaling**(i) * filters),
                conv_layers=block_conv_layers,
                activation=activation,
                **common_params
            ))
            self.decoding_layers.append(UnetDecodingLayer(
                kernel_size=main_kernel,
                filters=int(filter_scaling**(i) * filters),
                conv_layers=block_conv_layers,
                activation=activation,
                bilinear=upsampling_interpolation,
                **common_params
            ))
        # pre/post-bottleneck layer
        self.encoding_layers.append(UnetEncodingLayer(
            kernel_size=pre_bottle_kernel,
            filters=int(filter_scaling**(i+1) * filters),
            conv_layers=block_conv_layers,
            activation=activation,
            strides=bottle_stride,
            **common_params
        ))
        self.decoding_layers.append(UnetDecodingLayer(
            kernel_size=pre_bottle_kernel,
            filters=int(filter_scaling**(i+1) * filters),
            conv_layers=block_conv_layers,
            activation=activation,
            strides=bottle_stride,
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
            kernel_size=2*bottle_kernel,
            activation="linear",
            **common_params
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

    def call(self, kappa):
        skip_connections = []
        z = kappa
        for i in range(len(self.encoding_layers)):
            c_i, z = self.encoding_layers[i](z)
            skip_connections.append(c_i)
        skip_connections = skip_connections[::-1]
        z = self.bottleneck_layer1(z)
        z = self.bottleneck_layer2(z)
        for i in range(len(self.decoding_layers)):
            z = tf.concat([z, skip_connections[i]], axis=-1)
            z = self.encoding_layers[i](z)
        z = self.output_layer(z)
        return z

    def cost(self, kappa, alpha_true):
        alpha_pred = self.call(kappa)
        return tf.reduce_mean((alpha_pred - alpha_true) ** 2)
