import tensorflow as tf
from .layers import GatedConv, ConcatELU


class GatedConvNet(tf.keras.Model):
    def __init__(self, c_in, c_hidden, c_out=-1, num_layers=3):
        super().__init__()
        c_out = c_out if c_out > 0 else 2 * c_in
        layers = []
        layers += [GatedConv(c_hidden)]
        for i in range(num_layers):
            layers += [GatedConv(c_hidden), tf.keras.layers.LayerNormalization()]
        layers += [ConcatELU(), GatedConv(c_out)]
        self.call_layers = layers

    def __call__(self, x):
        for layer in self.call_layers:
            x = layer(x)
        return x