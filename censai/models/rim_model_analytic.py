import tensorflow as tf
from censai.definitions import DTYPE
from .layers import GRU
from .utils import get_activation


class ModelAnalytic(tf.keras.Model):
    def __init__(
            self,
            layers=2,
            units=32,
            unit_scaling=4,
            mlp_before_gru=2,
            activation="tanh",
            unit_cap=1024
        ):

        super(ModelAnalytic, self).__init__()
        activation = get_activation(activation)
        self.n_layers = layers
        self.hidden_units = units
        self.hidden_units_scaling = unit_scaling
        self.hidden_unit_cap = unit_cap
        self._feature_layers = []
        self._up_grus = []
        self._down_grus = []
        self._reconstruction_layers = []
        for i in range(layers):
            self._feature_layers.append([
                tf.keras.layers.Dense(
                    units=int(min(units * unit_scaling**(i + 1), unit_cap)),
                    activation=activation
                )
                for j in range(mlp_before_gru)]
            )
            self._up_grus.append(GRU(hidden_units=int(min(units * unit_scaling**(i + 1), unit_cap))))
            self._down_grus.append(GRU(hidden_units=int(min(units * unit_scaling**(i + 1), unit_cap))))
            self._reconstruction_layers.append([
                tf.keras.layers.Dense(
                    units=int(min(units * unit_scaling**(i + 1), unit_cap)),
                    activation=activation
                )
                for j in range(mlp_before_gru)]
            )
        self._reconstruction_layers = self._reconstruction_layers[::-1]
        self._down_grus = self._down_grus[::-1]

        self.output_layer = tf.keras.layers.Dense(units=13, activation="linear")

    def __call__(self, xt, states):
        return self.call(xt, states)

    def call(self, xt, states):
        new_states = []
        dxt = tf.identity(xt)
        for i, layers in enumerate(self._feature_layers):
            for layer in layers:
                dxt = layer(dxt)
            dxt, ht = self._down_grus[i](dxt, states[i])
            new_states.append(ht)
        for i, layers in enumerate(self._reconstruction_layers):
            for layer in layers:
                dxt = layer(dxt)
            dxt, ht = self._up_grus[i](dxt, states[i+self.n_layers])
            new_states.append(ht)
        dxt = self.output_layer(dxt)
        return dxt, new_states

    def init_hidden_states(self, batch_size):
        hts = []
        for i in range(self.n_layers):
            hts.append(tf.zeros(shape=(batch_size, int(min(self.hidden_units * self.hidden_units_scaling**(i + 1), self.hidden_unit_cap))), dtype=DTYPE))
        for i in reversed(range(self.n_layers)):
            hts.append(tf.zeros(shape=(batch_size, int(min(self.hidden_units * self.hidden_units_scaling**(i + 1), self.hidden_unit_cap))), dtype=DTYPE))
        return hts
