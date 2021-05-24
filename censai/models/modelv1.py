import tensorflow as tf
from censai.definitions import DTYPE
from censai.models.layers.conv_gru import ConvGRU
from censai.models.utils import global_step, summary_histograms


class Model(tf.keras.models.Model):
    def __init__(self,
                 name="modelv1",
                 kernel_size_downsampling=3,
                 filters_downsampling=16,
                 downsampling_layers=1,  # same for upsampling
                 conv_layers=1,
                 kernel_size_gru=3,
                 state_depth=64,
                 hidden_layers=1,
                 kernel_size_upsampling=3,
                 filters_upsampling=16,
                 kernel_regularizer_amp=0.,
                 bias_regularizer_amp=0.,
                 batch_norm=False,
                 dtype=DTYPE,
                 activation="leaky_relu",
                 **kwargs):
        super(Model, self).__init__(dtype=dtype, name=name, **kwargs)
        self._timestep_mod = 30  # silent instance attribute to be modified if needed in the RIM fit method
        self.downsampling_block = []
        self.recurrent_block = []
        self.upsampling_block = []
        self.hidden_conv = []
        if activation == "leaky_relu":
            activation = tf.keras.layers.LeakyReLU()
        elif activation == "gelu":
            activation = tf.keras.activations.gelu
        else:
            activation = tf.keras.layers.Activation(activation)
        for i in range(downsampling_layers):
            self.downsampling_block.append(tf.keras.layers.Conv2D(
                strides=2,
                kernel_size=kernel_size_downsampling,
                filters=filters_downsampling,
                name=f"DownsampleConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.downsampling_block.append(tf.keras.layers.BatchNormalization(name=f"BatchNormDownsample{i+1}", axis=-1))
            for j in range(conv_layers):
                if i == downsampling_layers - 1:
                    filters=state_depth//2 # match convGRU filter size
                else:
                    filters=filters_downsampling
                self.downsampling_block.append(tf.keras.layers.Conv2D(
                    strides=1,
                    kernel_size=kernel_size_downsampling,
                    filters=filters,
                    name=f"Conv{j + 1}",
                    activation=activation,
                    padding="same",
                    kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                    bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                    data_format="channels_last",
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                ))
                if batch_norm:
                    self.downsampling_block.append(
                        tf.keras.layers.BatchNormalization(name=f"BatchNormDownsampleConv{j + 1}", axis=-1))
        for i in range(downsampling_layers):
            self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                strides=2,
                kernel_size=kernel_size_upsampling,
                filters=filters_upsampling,
                name=f"UpsampleConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.upsampling_block.append(tf.keras.layers.BatchNormalization(name=f"BatchNormUpsample{i+1}", axis=-1))
            for j in range(conv_layers):
                self.upsampling_block.append(tf.keras.layers.Conv2DTranspose(
                    strides=1,
                    kernel_size=kernel_size_upsampling,
                    filters=filters_upsampling,
                    name=f"TConv{j + 1}",
                    activation=activation,
                    padding="same",
                    kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                    bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                    data_format="channels_last",
                    kernel_initializer=tf.keras.initializers.GlorotUniform()
                ))
                if batch_norm and (j != conv_layers-1 or i != downsampling_layers-1): # except last layer
                    self.upsampling_block.append(
                        tf.keras.layers.BatchNormalization(name=f"BatchNormUpsampleConv{j + 1}", axis=-1))
        self.gru1 = ConvGRU(filters=state_depth//2, kernel_size=kernel_size_gru)
        self.gru2 = ConvGRU(filters=state_depth//2, kernel_size=kernel_size_gru)
        for i in range(hidden_layers):
            self.hidden_conv.append(tf.keras.layers.Conv2D(
                filters=state_depth//2,
                kernel_size=kernel_size_gru,
                name=f"HiddenConv{i+1}",
                activation=activation,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_regularizer_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_regularizer_amp),
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            if batch_norm:
                self.hidden_conv.append(
                    tf.keras.layers.BatchNormalization(name=f"BatchNormHiddenConv{i + 1}", axis=-1))
        self.output_layer = tf.keras.layers.Conv2DTranspose(
            name="output_conv",
            kernel_size=1,
            filters=1,
            activation=tf.keras.layers.Activation("linear"),
            padding="same",
        )

    def call(self, X, ht):
        """
        :param yt: Image tensor of shape [batch, pixel, pixel, channel], correspond to the step t of the reconstruction.
        :param ht: Hidden memory tensor updated in the Recurrent Block
        """
        for layer in self.downsampling_block:
            X = layer(X)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, X)

        # ===== Recurrent Block =====
        ht_1, ht_2 = tf.split(ht, 2, axis=3)
        ht_1 = self.gru1(X, ht_1)  # to be recombined in new state
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru1, ht_1)
        ht_1_features = tf.identity(ht_1)
        for layer in self.hidden_conv:
            ht_1_features = layer(ht_1_features)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, ht_1_features)
        ht_2 = self.gru2(ht_1_features, ht_2)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.gru2, ht_2)
        # ===========================

        delta_xt = tf.identity(ht_2)
        for layer in self.upsampling_block:
            delta_xt = layer(delta_xt)
            if global_step() % self._timestep_mod == 0:
                summary_histograms(layer, delta_xt)
        delta_xt = self.output_layer(delta_xt)
        if global_step() % self._timestep_mod == 0:
            summary_histograms(self.output_layer, delta_xt)
        new_state = tf.concat([ht_1, ht_2], axis=3)
        return delta_xt, new_state
