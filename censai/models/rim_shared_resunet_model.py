import tensorflow as tf
from censai.models.layers import ResUnetDecodingLayer, ResUnetEncodingLayer, ConvGRUBlock, ConvGRUPlusBlock
from .utils import get_activation
from censai.definitions import DTYPE


class SharedResUnetModel(tf.keras.Model):
    def __init__(
            self,
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
            input_kernel_size=11,
            gru_kernel_size=None,
            group_norm=False,
            dropout_rate=None,
            upsampling_interpolation=False,  # use strided transposed convolution if false
            kernel_l1_amp=0.,
            bias_l1_amp=0.,
            kernel_l2_amp=0.,
            bias_l2_amp=0.,
            activation="leaky_relu",
            alpha=0.1,                       # for leaky relu
            use_bias=True,
            trainable=True,
            gru_architecture="concat",  # or "plus"
            initializer="glorot_uniform",
    ):
        super(SharedResUnetModel, self).__init__(name=name)
        self.trainable = trainable

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L1L2(l1=kernel_l1_amp, l2=kernel_l2_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L1L2(l1=bias_l1_amp, l2=bias_l2_amp)})

        kernel_size = (kernel_size,)*2
        resampling_kernel_size = resampling_kernel_size if resampling_kernel_size is not None else kernel_size
        bottleneck_kernel_size = bottleneck_kernel_size if bottleneck_kernel_size is not None else kernel_size
        bottleneck_filters = bottleneck_filters if bottleneck_filters is not None else int(filter_scaling**layers  * filters)
        gru_kernel_size = gru_kernel_size if gru_kernel_size is not None else kernel_size
        activation = get_activation(activation, alpha=alpha)
        GRU = ConvGRUBlock if gru_architecture == "concat" else ConvGRUPlusBlock

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
                ResUnetEncodingLayer(
                    kernel_size=kernel_size,
                    downsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    downsampling_filters=int(filter_scaling**(i+1) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    group_norm=group_norm,
                    groups=min(1, int(filter_scaling ** (i) * filters) // 8),
                    dropout_rate=dropout_rate,
                    **common_params
                )
            )
            self.decoding_layers.append(
                ResUnetDecodingLayer(
                    kernel_size=kernel_size,
                    upsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    bilinear=upsampling_interpolation,
                    group_norm=group_norm,
                    groups=min(1, int(filter_scaling ** (i) * filters) // 8),
                    dropout_rate=dropout_rate,
                    **common_params
                )
            )
            self.gated_recurrent_blocks.append(
                    GRU(
                        filters=int(filter_scaling**(i) * filters),
                        kernel_size=gru_kernel_size,
                        activation=activation
                )
            )

        self.decoding_layers = self.decoding_layers[::-1]

        self.bottleneck_gru = GRU(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation
        )

        self.output_layer = tf.keras.layers.Conv2D(
            filters=2,  # source and kappa
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

        self.input_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=input_kernel_size,
            **common_params
        )

    def __call__(self, source, kappa, source_grad, kappa_grad, states):
        return self.call(source, kappa, source_grad, kappa_grad, states)

    def call(self, source, kappa, source_grad, kappa_grad, states):
        delta_xt = tf.concat([source, source_grad, kappa, kappa_grad], axis=-1)
        delta_xt = self.input_layer(delta_xt)
        skip_connections = []
        new_states = []
        for i in range(self._num_layers):
            c_i, delta_xt = self.encoding_layers[i](delta_xt)
            c_i, new_state = self.gated_recurrent_blocks[i](c_i, states[i])  # Pass skip connections through GRU and update states
            skip_connections.append(c_i)
            new_states.append(new_state)
        skip_connections = skip_connections[::-1]
        delta_xt, new_state = self.bottleneck_gru(delta_xt, states[-1])
        new_states.append(new_state)
        for i in range(self._num_layers):
            delta_xt = self.decoding_layers[i](delta_xt, skip_connections[i])
        delta_xt = self.output_layer(delta_xt)
        source_delta, kappa_delta = tf.split(delta_xt, 2, axis=-1)
        new_source = source + source_delta
        new_kappa = kappa + kappa_delta
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

