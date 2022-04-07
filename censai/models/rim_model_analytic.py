import tensorflow as tf
from .layers import GRU
from .utils import get_activation


class ModelAnalytic(tf.keras.Model):
    def __init__(
            self,
            layers=2,  # before and after GRU
            units=32,
            unit_scaling=1,
            activation="tanh"
        ):

        super(ModelAnalytic, self).__init__()
        activation = get_activation(activation)
        self.n_layers = layers
        self.hidden_units = units
        self.hidden_units_scaling = unit_scaling
        self._feature_layers = []
        self._up_grus = []
        self._down_grus = []
        self._reconstruction_layers = []

        for i in range(layers):
            self._feature_layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    activation=activation
                )
            )
            self._reconstruction_layers.append(
                tf.keras.layers.Dense(
                    units=units,
                    activation=activation
                )
            )
        self._reconstruction_layers = self._reconstruction_layers[::-1]
        self.gru = GRU(hidden_units=units)

        self.output_layer = tf.keras.layers.Dense(units=13, activation="linear")

    def __call__(self, xt, states):
        return self.call(xt, states)

    def call(self, xt, states):
        dxt = tf.identity(xt)
        for layer in self._feature_layers:
            dxt = layer(dxt)
        dxt, new_state = self.gru(dxt, states)
        for layer in self._reconstruction_layers:
            dxt = layer(dxt)
        dxt = self.output_layer(dxt)
        return dxt, new_state

    def init_hidden_states(self, batch_size):
        return tf.zeros(shape=(batch_size, self.hidden_units))
