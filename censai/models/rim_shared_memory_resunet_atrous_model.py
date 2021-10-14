import tensorflow as tf
from censai.models.layers import ResUnetAtrousEncodingLayer, ResUnetAtrousDecodingLayer, ConvGRUBlock, ConvGRUPlusBlock, PSP
from .utils import get_activation
from censai.definitions import DTYPE


class SharedMemoryResUnetAtrousModel(tf.keras.Model):
    def __init__(
            self,
            dilation_rates:list,
            pixels:int,
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
            input_kernel_size=1,
            gru_kernel_size=None,
            group_norm=True,
            dropout_rate=None,
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
            psp_bottleneck=True,
            psp_output=True,
            psp_scaling=2
    ):
        super(SharedMemoryResUnetAtrousModel, self).__init__(name=name)
        self.trainable = trainable

        common_params = {"padding": "same", "kernel_initializer": initializer,
                         "data_format": "channels_last", "use_bias": use_bias,
                         "kernel_regularizer": tf.keras.regularizers.L1L2(l1=kernel_l1_amp, l2=kernel_l2_amp)}
        if use_bias:
            common_params.update({"bias_regularizer": tf.keras.regularizers.L1L2(l1=bias_l1_amp, l2=bias_l2_amp)})

        resampling_kernel_size = resampling_kernel_size if resampling_kernel_size is not None else kernel_size
        bottleneck_kernel_size = bottleneck_kernel_size if bottleneck_kernel_size is not None else kernel_size
        bottleneck_filters = bottleneck_filters if bottleneck_filters is not None else 2*int(filter_scaling**layers * filters)
        gru_kernel_size = gru_kernel_size if gru_kernel_size is not None else kernel_size
        activation = get_activation(activation, alpha=alpha)
        GRU = ConvGRUBlock if gru_architecture == "concat" else ConvGRUPlusBlock

        self._num_layers = layers
        self._strides = strides
        self._init_filters = filters
        self._filter_scaling = filter_scaling
        self._bottleneck_filters = bottleneck_filters

        self.kappa_encoding_layers = []
        self.kappa_decoding_layers = []
        self.source_encoding_layers = []
        self.source_decoding_layers = []
        self.gated_recurrent_blocks = []
        self.skip_connection_combine = []
        self.skip_connection_group_norm = []
        for i in range(layers):
            self.kappa_encoding_layers.append(
                ResUnetAtrousEncodingLayer(
                    kernel_size=kernel_size,
                    downsampling_kernel_size=resampling_kernel_size,
                    downsampling_filters=int(filter_scaling ** (i + 1) * filters),
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    group_norm=group_norm,
                    dropout_rate=dropout_rate,
                    dilation_rates=dilation_rates[i],
                    **common_params
                )
            )
            self.source_encoding_layers.append(
                ResUnetAtrousEncodingLayer(
                    kernel_size=kernel_size,
                    downsampling_kernel_size=resampling_kernel_size,
                    downsampling_filters=int(filter_scaling ** (i + 1) * filters),
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    strides=strides,
                    group_norm=group_norm,
                    groups=min(1, int(filter_scaling**(i) * filters)//8),
                    dropout_rate=dropout_rate,
                    dilation_rates=dilation_rates[i],
                    **common_params
                )
            )
            self.kappa_decoding_layers.append(
                ResUnetAtrousDecodingLayer(
                    kernel_size=kernel_size,
                    upsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    group_norm=group_norm,
                    groups=min(1, int(filter_scaling**(i) * filters)//8),
                    dropout_rate=dropout_rate,
                    dilation_rates=dilation_rates[i],
                    **common_params
                )
            )
            self.source_decoding_layers.append(
                ResUnetAtrousDecodingLayer(
                    kernel_size=kernel_size,
                    upsampling_kernel_size=resampling_kernel_size,
                    filters=int(filter_scaling**(i) * filters),
                    conv_layers=block_conv_layers,
                    activation=activation,
                    group_norm=group_norm,
                    groups=min(1, int(filter_scaling ** (i) * filters) // 8),
                    dropout_rate=dropout_rate,
                    dilation_rates=dilation_rates[i],
                    **common_params
                )
            )
            self.skip_connection_combine.append(tf.keras.layers.Conv2D(
                filters=int(filter_scaling**(i) * filters),
                kernel_size=1,
                **common_params
            ))
            self.skip_connection_group_norm.append(tf.keras.layers.BatchNormalization() if group_norm else tf.identity)
            self.gated_recurrent_blocks.append(
                    GRU(
                        filters=int(filter_scaling**(i) * filters),
                        kernel_size=gru_kernel_size,
                        activation=activation
                )
            )

        self.kappa_decoding_layers = self.kappa_decoding_layers[::-1]
        self.source_decoding_layers = self.source_decoding_layers[::-1]

        self.bottleneck_combine = tf.keras.layers.Conv2D(
            filters=bottleneck_filters,
            kernel_size=1
        )
        self.bottleneck_group_norm = tf.keras.layers.BatchNormalization() if group_norm else tf.identity
        self.bottleneck_gru = GRU(
            filters=bottleneck_filters,
            kernel_size=bottleneck_kernel_size,
            activation=activation
        )

        self.source_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )
        self.kappa_output_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation="linear",
            **common_params
        )

        self.source_input_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=input_kernel_size,
            **common_params
        )
        self.kappa_input_layer = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=input_kernel_size,
            **common_params
        )

        if psp_bottleneck:
            self.source_psp_bottleneck = PSP(filters=int(filter_scaling**layers * filters), pixels=pixels//strides**layers,  scaling=psp_scaling, bilinear=False, group_norm=group_norm)
            self.kappa_psp_bottleneck  = PSP(filters=int(filter_scaling**layers * filters), pixels=pixels//strides**layers,  scaling=psp_scaling, bilinear=False, group_norm=group_norm)
        else:
            self.source_psp_bottleneck = tf.identity
            self.kappa_psp_bottleneck = tf.identity

        if psp_output:
            self.source_psp_output = PSP(filters=filters, pixels=pixels, scaling=psp_scaling, bilinear=True)
            self.kappa_psp_output = PSP(filters=filters, pixels=pixels, scaling=psp_scaling, bilinear=True)
        else:
            self.source_psp_output = tf.identity
            self.kappa_psp_output = tf.identity

    def __call__(self, source, kappa, source_grad, kappa_grad, states):
        return self.call(source, kappa, source_grad, kappa_grad, states)

    def call(self, source, kappa, source_grad, kappa_grad, states):
        source_delta = tf.concat([tf.identity(source), source_grad], axis=-1)
        kappa_delta = tf.concat([tf.identity(kappa), kappa_grad], axis=-1)
        source_delta = self.source_input_layer(source_delta)
        kappa_delta = self.kappa_input_layer(kappa_delta)

        skip_connections = []
        new_states = []
        for i in range(self._num_layers):
            s_i, source_delta = self.source_encoding_layers[i](source_delta)
            k_i, kappa_delta = self.kappa_encoding_layers[i](kappa_delta)
            c_i = self.skip_connection_combine[i](tf.concat([s_i, k_i], axis=-1))
            c_i = self.skip_connection_group_norm[i](c_i)
            c_i, new_state = self.gated_recurrent_blocks[i](c_i, states[i])  # Pass skip connections through GRU and update states
            skip_connections.append(c_i)
            new_states.append(new_state)
        skip_connections = skip_connections[::-1]
        source_delta = self.source_psp_bottleneck(source_delta)
        kappa_delta = self.kappa_psp_bottleneck(kappa_delta)
        c_bottleneck = self.bottleneck_combine(tf.concat([source_delta, kappa_delta], axis=-1))
        c_bottleneck = self.bottleneck_group_norm(c_bottleneck)
        delta_xt, new_state = self.bottleneck_gru(c_bottleneck, states[-1])
        source_delta, kappa_delta = tf.split(delta_xt, 2, axis=-1)
        new_states.append(new_state)
        for i in range(self._num_layers):
            source_delta = self.source_decoding_layers[i](source_delta, skip_connections[i])
            kappa_delta = self.source_decoding_layers[i](kappa_delta, skip_connections[i])
        source_delta = self.source_psp_output(source_delta)
        kappa_delta = self.kappa_psp_output(kappa_delta)
        source_delta = self.source_output_layer(source_delta)
        kappa_delta = self.kappa_output_layer(kappa_delta)
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

