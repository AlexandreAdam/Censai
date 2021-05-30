import tensorflow as tf
from .layers.resnet_block import ResidualBlock


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()


    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def forward_pass(self, x):
        return self.decode(self.encode(x))

    def cost_function(self, x):
        return x

