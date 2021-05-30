import tensorflow as tf
from censai.definitions import DTYPE


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 conv_layers=2,
                 kernel_reg_amp=0.01,
                 bias_reg_amp=0.01,
                 alpha=0.1,
                 dropout_rate=None,
                 architecture="bare",
                 **kwargs):
        super(ResidualBlock, self).__init__(DTYPE, **kwargs)
        assert architecture in ["bare", "original", "bn_after_addition",
                                "relu_before_addition", "relu_only_pre_activation",
                                "full_pre_activation", "full_pre_activation_rescale"]
        self.conv_layers = []
        self.non_linearity = tf.keras.layers.LeakyReLU(alpha=alpha)
        if architecture == "full_pre_activation_rescale":
            filters = filters//2
        for i in range(conv_layers):
            self.conv_layers.append(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=1,
                    padding="same",
                    data_format="channels_last",
                    kernel_initializer=tf.keras.initializers.HeUniform(),
                    kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                    bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
                )
            )
        if dropout_rate is not None:
            assert isinstance(dropout_rate, float)
            assert dropout_rate < 1
            assert dropout_rate >= 0
            self.dropout = tf.keras.layers.SpatialDropout2D(dropout_rate, data_format="channels_last")
        else:
            self.dropout = tf.identity
        if architecture != "bare":
            self.batch_norms = []
            if architecture != "full_pre_activation_rescale":
                for i in range(conv_layers):
                    self.batch_norms.append(
                        tf.keras.layers.BatchNormalization()
                    )
            else:
                for i in range(conv_layers + 2):
                    self.batch_norms.append(
                        tf.keras.layers.BatchNormalization()
                    )
        if architecture == "full_pre_activation_rescale":
            self.rescale_input = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=1,
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.HeUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
            )
            self.rescale_output = tf.keras.layers.Conv2D(
                filters=filters*2,
                kernel_size=1,
                strides=1,
                padding="same",
                data_format="channels_last",
                kernel_initializer=tf.keras.initializers.HeUniform(),
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp)
            )

        self._call = {
            "bare": _bare,
            "original": _original,
            "bn_after_addition": _bn_after_addition,
            "relu_before_addition": _relu_before_addition,
            "relu_only_pre_activation": _relu_only_pre_activation,
            "full_pre_activation": _full_pre_activation,
            "full_pre_activation_rescale": _full_pre_activation_rescale
        }[architecture]

    def call(self, x):
        return self._call(self, x)

@tf.function
def _bare(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for layer in res_block.conv_layers:
        features = layer(features)
        features = res_block.non_linearity(features)
        features = res_block.dropout(features)
    return tf.add(x, features)


@tf.function
def _original(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for i, layer in enumerate(res_block.conv_layers):
        features = layer(features)
        features = res_block.batch_norms[i](features)
        if i != len(res_block.conv_layers) - 1:
            features = res_block.non_linearity(features)
            features = res_block.dropout(features)
    features = tf.add(x, features)
    features = res_block.non_linearity(features)
    return features


@tf.function
def _bn_after_addition(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for i, layer in enumerate(res_block.conv_layers):
        features = layer(features)
        if i != len(res_block.conv_layers) - 1:
            features = res_block.batch_norms[i](features)
            features = res_block.non_linearity(features)
            features = res_block.dropout(features)
    features = tf.add(x, features)
    features = res_block.batch_norms[-1](features)
    features = res_block.non_linearity(features)
    return features


@tf.function
def _relu_before_addition(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for i, layer in enumerate(res_block.conv_layers):
        features = layer(features)
        features = res_block.batch_norms[i](features)
        features = res_block.non_linearity(features)
        features = res_block.dropout(features)
    features = tf.add(x, features)
    return features


@tf.function
def _relu_only_pre_activation(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for i, layer in enumerate(res_block.conv_layers):
        features = res_block.non_linearity(features)
        features = layer(features)
        features = res_block.batch_norms[i](features)
        features = res_block.dropout(features)
    features = tf.add(x, features)
    return features


@tf.function
def _full_pre_activation(res_block: ResidualBlock, x):
    features = tf.identity(x)
    for i, layer in enumerate(res_block.conv_layers):
        features = res_block.batch_norms[i](features)
        features = res_block.non_linearity(features)
        features = layer(features)
        features = res_block.dropout(features)
    features = tf.add(x, features)
    return features


@tf.function
def _full_pre_activation_rescale(res_block: ResidualBlock, x):
    features = tf.identity(x)
    # One by One Conv rescale
    features = res_block.batch_norms[0](features)
    features = res_block.non_linearity(features)
    features = res_block.rescale_input(features)
    for i, layer in enumerate(res_block.conv_layers):
        features = res_block.batch_norms[i+1](features)
        features = res_block.non_linearity(features)
        features = layer(features)
        features = res_block.dropout(features)
    # rescale output, recover initial filters
    features = res_block.batch_norms[-1](features)
    features = res_block.non_linearity(features)
    features = res_block.rescale_output(features)
    features = tf.add(x, features)
    return features
