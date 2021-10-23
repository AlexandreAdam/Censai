import tensorflow as tf
from .layers import GatedConv, ConcatELU


class GatedConvNet(tf.keras.layers.Layer):
    def __init__(self, filters, num_layers=3, filters_out=-1):
        super().__init__()
        c_out = filters_out if filters_out > 0 else 2 * filters
        layers = []
        layers += [GatedConv(filters)]
        for i in range(num_layers):
            layers += [GatedConv(filters), tf.keras.layers.LayerNormalization()]
        layers += [ConcatELU(), GatedConv(c_out)]
        self.call_layers = layers

    def __call__(self, x):
        for layer in self.call_layers:
            x = layer(x)
        return x