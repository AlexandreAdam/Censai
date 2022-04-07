import tensorflow as tf
from censai.models.utils import get_activation


class ModelCNNAnalytic(tf.keras.Model):
    """
    Incorporate the function F^{-1}(y) in the model
    """
    def __init__(
            self,
            levels=4,
            layer_per_level=2,
            output_features=13,
            kernel_size=3,
            input_kernel_size=11,
            strides=2,
            filters=32,
            filter_scaling=1,
            filter_cap=1024,
            activation="tanh"
        ):

        super(ModelCNNAnalytic, self).__init__()
        activation = get_activation(activation)
        self._feature_layers = []
        self.input_layer = tf.keras.layers.Conv2D(
            kernel_size=input_kernel_size,
            filters=filters,
            padding="same",
            activation=activation
        )
        for i in range(levels):
            self._feature_layers.extend([
                tf.keras.layers.Conv2D(
                    kernel_size=kernel_size,
                    filters=max(int(filters * filter_scaling ** i), filter_cap),
                    padding="same",
                    activation=activation
                )
            for j in range(layer_per_level)]
            )
            self._feature_layers.append(
                tf.keras.layers.Conv2D(
                    kernel_size=kernel_size,
                    filters=max(int(filters * filter_scaling ** i), filter_cap),
                    padding="same",
                    activation=activation,
                    strides=strides
                )
            )

        self.flatten = tf.keras.layers.Flatten(data_format="channels_last")
        self.output_layer = tf.keras.layers.Dense(units=output_features, activation="linear")

    def __call__(self, y):
        return self.call(y)

    def call(self, y):
        x = tf.identity(y)
        for layer in self._feature_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    y = tf.random.normal(shape=[1, 128, 128, 1])
    model = ModelCNNAnalytic()
    print(model.call(y))