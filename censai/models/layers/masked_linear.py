import tensorflow as tf


class MaskedLinear(tf.keras.layers.Layer):
    """ MADE building block layer """
    def __init__(self, units, mask, cond_label_size=None):
        super().__init__()
        self.mask = mask
        self._layer = tf.keras.layers.Dense(units)

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)