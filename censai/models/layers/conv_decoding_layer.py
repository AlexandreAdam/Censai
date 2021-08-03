import tensorflow as tf
from censai.models.utils import get_activation


class ConvDecodingLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size=3,
            upsampling_kernel_size=None,
            filters=32,
            conv_layers=2,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            name=None,
            strides=2,
            bilinear=False,
            kernel_reg_amp=1e-4,
            bias_reg_amp=1e-4
    ):
        super(ConvDecodingLayer, self).__init__(name=name)
        if upsampling_kernel_size is None:
            self.upsampling_kernel_size = kernel_size
        else:
            self.upsampling_kernel_size = tuple([upsampling_kernel_size]*2)
        self.kernel_size = tuple([kernel_size]*2)
        self.num_conv_layers = conv_layers
        self.filters = filters
        self.strides = tuple([strides]*2)
        self.activation = get_activation(activation)

        self.conv_layers = []
        self.batch_norms = []
        for i in range(self.num_conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    padding="SAME",
                    data_format="channels_last",
                    kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
                    bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp),
                )
            )
            if batch_norm:
                self.batch_norms.append(
                    tf.keras.layers.BatchNormalization()
                )
            else:
                self.batch_norms.append(
                    tf.identity
                )
        if bilinear:
            self.upsampling_layer = tf.keras.layers.UpSampling2D(size=self.strides, interpolation="bilinear")
        else:
            self.upsampling_layer = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2DTranspose(
                        filters=self.filters,
                        kernel_size=self.upsampling_kernel_size,
                        strides=self.strides,
                        padding="SAME",
                        data_format="channels_last",
                        kernel_regularizer=tf.keras.regularizers.l2(kernel_reg_amp),
                        bias_regularizer=tf.keras.regularizers.l2(bias_reg_amp)
                    ),
                    tf.keras.layers.BatchNormalization() if batch_norm else tf.keras.layers.Lambda(lambda x: tf.identity(x)),
                    self.activation
                ]
            )

        if dropout_rate is None:
            self.dropout = tf.identity
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x):
        x = self.upsampling_layer(x)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

    def call_with_skip_connection(self, x, skip_connection):
        x = self.upsampling_layer(x)
        x = tf.add(x, skip_connection)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        return x