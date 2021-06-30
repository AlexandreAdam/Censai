import tensorflow as tf
from censai.models.layers import UnetDecodingLayer, UnetEncodingLayer, DownsamplingLayer, UpsamplingLayer
from .layers.conv_gru_component import ConvGRUBlock
from .utils import get_activation
from censai.definitions import DTYPE


class SharedUnetModel(tf.keras.Model):
    def __init__(
            self,
            kappa_resize_layers,
            name="RIMUnetModel",
            filters=32,
            filter_scaling=1,
            kernel_size=3,
            layers=2,                        # before bottleneck
            block_conv_layers=2,
            strides=2,
            bottleneck_kernel_size=None,     # use kernel_size as default
            bottleneck_filters=None,
            resampling_kernel_size=None,
            gru_kernel_size=None,
            upsampling_interpolation=False,  # use strided transposed convolution if false
            kernel_regularizer_amp=0.,
            bias_regularizer_amp=0.,
            activation="leaky_relu",
            alpha=0.1,                       # for leaky relu
            use_bias=True,
            trainable=True,
            initializer="glorot_uniform",
            kappa_resize_separate_grad_downsampling=False,
            kappa_resize_method="bilinear",
            kappa_resize_filters=4,
            kappa_resize_kernel_size=3,
            kappa_resize_conv_layers=1,
            kappa_resize_strides=2
    ):
        super(SharedUnetModel, self).__init__(name=name)
        self.trainable = trainable

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L2(l2=kernel_regularizer_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L2(l2=bias_regularizer_amp)})

        resampling_kernel_size = resampling_kernel_size if resampling_kernel_size is not None else kernel_size
        bottleneck_kernel_size = bottleneck_kernel_size if bottleneck_kernel_size is not None else kernel_size
        bottleneck_filters = bottleneck_filters if bottleneck_filters is not None else int(filter_scaling**(layers + 1) * filters)
        gru_kernel_size = gru_kernel_size if gru_kernel_size is not None else kernel_size
        activation = get_activation(activation, alpha=alpha)

        self._num_layers = layers
        self._strides = strides
        self._init_filters = filters
        self._filter_scaling = filter_scaling
        self._bottleneck_filters = bottleneck_filters

        self.encoding_layers = []
        self.decoding_layers = []
        self.gated_recurrent_blocks = []
        for i in range(layers):
            self.encoding_layers.append(
                UnetEncodingLayer(
                    kernel_size=kernel_size,
                    downsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    **common_params
                )
            )
            self.decoding_layers.append(
                UnetDecodingLayer(
                    kernel_size=kernel_size,
                    upsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    bilinear=upsampling_interpolation,
                    **common_params
                )
            )
            self.gated_recurrent_blocks.append(
                    ConvGRUBlock(
                        filters=2*int(filter_scaling**(i) * filters),
                        kernel_size=gru_kernel_size,
                        activation=activation
                )
            )

        self.decoding_layers = self.decoding_layers[::-1]

        self.bottleneck_layer1 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation,
            **common_params
        )
        self.bottleneck_layer2 = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation,
            **common_params
        )
        self.bottleneck_gru = ConvGRUBlock(
            filters=2*bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=2,  # source and kappa
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

        # Some logic here to support 2 modes of downsampling
        self.separate_grad_downsampling = kappa_resize_separate_grad_downsampling
        if kappa_resize_layers == 0:
            self.upsampling_layer = tf.identity
            self.downsampling_layer = tf.identity
            self.grad_downsampling_layer = tf.identity
        else:
            self.upsampling_layer = UpsamplingLayer(
                method=kappa_resize_method,
                layers=kappa_resize_layers,
                conv_layers=kappa_resize_conv_layers,
                kernel_size=kappa_resize_kernel_size,
                strides=kappa_resize_strides,
                filters=kappa_resize_filters,
                output_filters=1
            )
            self.downsampling_layer = DownsamplingLayer(
                method=kappa_resize_method,
                layers=kappa_resize_layers,
                conv_layers=kappa_resize_conv_layers,
                kernel_size=kappa_resize_kernel_size,
                strides=kappa_resize_strides,
                filters=kappa_resize_filters,
                output_filters=1 if self.separate_grad_downsampling else 2
            )
            if self.separate_grad_downsampling:
                self.grad_downsampling_layer = DownsamplingLayer(
                    method=kappa_resize_method,
                    layers=kappa_resize_layers,
                    conv_layers=kappa_resize_conv_layers,
                    kernel_size=kappa_resize_kernel_size,
                    strides=kappa_resize_strides,
                    filters=kappa_resize_filters,
                    output_filters=1
                )
            else:
                self.grad_downsampling_layer = None

    def __call__(self, source, kappa, source_grad, kappa_grad, states):
        return self.call(source, kappa, source_grad, kappa_grad, states)

    def call(self, source, kappa, source_grad, kappa_grad, states):
        if not self.separate_grad_downsampling:
            kappa, kappa_grad = tf.split(
                self.downsampling_layer(
                    tf.concat([kappa, kappa_grad], axis=-1)
                ),
                2, axis=-1
            )
        else:
            kappa = self.downsampling_layer(kappa)
            kappa_grad = self.grad_downsampling_layer(kappa_grad)
        delta_xt = tf.concat([source, source_grad, kappa, kappa_grad], axis=-1)
        skip_connections = []
        for i in range(len(self.encoding_layers)):
            c_i, delta_xt = self.encoding_layers[i](delta_xt)
            skip_connections.append(c_i)
        delta_xt = self.bottleneck_layer1(delta_xt)
        delta_xt = self.bottleneck_layer2(delta_xt)
        # Pass skip connections through GRUS and update states
        new_states = []
        for i in range(len(self.gated_recurrent_blocks)):
            c_i, new_state = self.gated_recurrent_blocks[i](skip_connections[i], states[i])
            skip_connections[i] = c_i
            new_states.append(new_state)
        c_b, new_state = self.bottleneck_gru(delta_xt, states[-1])
        skip_connections.append(c_b)
        new_states.append(new_state)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoding_layers)):
            delta_xt = tf.concat([delta_xt, skip_connections[i]], axis=-1)
            delta_xt = self.decoding_layers[i](delta_xt)
        delta_xt = self.output_layer(delta_xt)

        source_delta, kappa_delta = tf.split(delta_xt, 2, axis=-1)
        new_source = source + source_delta
        new_kappa = kappa + kappa_delta
        new_kappa = self.upsampling_layer(new_kappa)
        return new_source, new_kappa, new_states

    def init_hidden_states(self, input_pixels, batch_size, constant=0.):
        hidden_states = []
        for i in range(self._num_layers):
            pixels = input_pixels // self._strides**(i)
            filters = int(self._filter_scaling**(i) * self._init_filters)
            hidden_states.append(
                constant * tf.ones(shape=[batch_size, pixels, pixels, 2 * filters], dtype=DTYPE)
            )
        pixels = input_pixels // self._strides ** (self._num_layers)
        hidden_states.append(
            constant * tf.ones(shape=[batch_size, pixels, pixels, 2 * self._bottleneck_filters], dtype=DTYPE)
        )
        return hidden_states

