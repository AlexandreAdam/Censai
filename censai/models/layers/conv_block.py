import tensorflow as tf
from censai.models.utils import get_activation


class ConvBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            kernel_size=3,
            filters=32,
            conv_layers=2,
            activation="linear",
            batch_norm=False,
            dropout_rate=None,
            name=None,
            kernel_reg_amp=0.01,
            bias_reg_amp=0.01,
    ):
        super(ConvBlock, self).__init__(name=name)
        self.kernel_size = tuple([kernel_size]*2)
        self.num_conv_layers = conv_layers
        self.filters = filters
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

        if dropout_rate is None:
            self.dropout = tf.identity
        else:
            self.dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate, data_format="channels_last")

    def call(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        return x

