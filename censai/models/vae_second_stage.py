import tensorflow as tf
from .utils import get_activation


class NN(tf.keras.Model):
    def __init__(
            self,
            output_size,
            hidden_layers=2,
            units=64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(),
            bias_regularizer=tf.keras.regularizers.l2()
    ):
        super(NN, self).__init__()
        if not isinstance(units, list):
            units = [units] * hidden_layers
        self.activitation = get_activation(activation)
        self.hidden_layers = []
        for i in range(hidden_layers):
            self.hidden_layers.append(tf.keras.layers.Dense(
                units=units[i],
                activation=self.activitation,
                kernel_initializer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            ))
        self.output_layer = tf.keras.layers.Dense(
            units=output_size,
            kernel_initializer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class VAESecondStage(tf.keras.Model):
    def __init__(
            self,
            latent_size,
            output_size,
            units=64,
            hidden_layers=2,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(),
            bias_regularizer=tf.keras.regularizers.l2()
    ):
        super(VAESecondStage, self).__init__()
        self.latent_size = latent_size
        self.encoder = NN(output_size=2 * latent_size, units=units, hidden_layers=hidden_layers, activation=activation,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
        self.decoder = NN(output_size=output_size, units=units, hidden_layers=hidden_layers, activation=activation,
                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)

    def encode(self, x):
        batch_size = x.shape[0]
        code = self.encoder(x)
        mean, logvar = tf.split(code, 2, axis=-1)
        logvar *= 0.5
        epsilon = tf.random.normal(shape=[batch_size, self.latent_size])
        z = mean + tf.exp(logvar) * epsilon
        return z, mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n_samples):
        z = tf.random.normal(shape=[n_samples, self.latent_size])
        return self.decode(z)

    def __call__(self, x):
        return self.call(x)

    def call(self, x):
        z, mean, logvar = self.encode(x)
        return self.decode(z)

    def cost_function(self, x):
        z, mean, logvar = self.encode(x)
        y = self.decode(z)
        reconstruction_cost = tf.reduce_sum((y - x)**2, axis=(1, 2, 3))
        kl_divergence = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=1)
        return reconstruction_cost, kl_divergence
