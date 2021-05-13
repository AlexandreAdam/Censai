import tensorflow as tf
from censai.definitions import lrelu4p


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        activation = lrelu4p
        D01 = tf.keras.layers.Conv2D(32, (6, 6), activation=activation, strides=2, padding='same')
        D02 = tf.keras.layers.Conv2D(32, (6, 6), activation=activation, strides=2, padding='same')
        D03 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, strides=2, padding='same')
        D04 = tf.keras.layers.Conv2D(1, (3, 3), activation=activation, strides=1, padding='same')
        self.encode_layers = [D01, D02, D03, D04]

        U01 = tf.keras.layers.Conv2DTranspose(16, (6, 6), activation=activation, strides=2, padding='same')
        U02 = tf.keras.layers.Conv2DTranspose(32, (6, 6), activation=activation, strides=2, padding='same')
        U03 = tf.keras.layers.Conv2DTranspose(16, (2, 2), activation=activation, strides=2, padding='same')
        U04 = tf.keras.layers.Conv2DTranspose(1, (2, 2), activation=activation, strides=1, padding='same')
        self.decode_layers = [U01, U02, U03, U04]
        self.AE_checkpoint_path = "checkpoints/AE_weights"

    def encode(self, X):
        for layer in self.encode_layers:
            X = layer(X)
        return X

    def decode(self, X):
        for layer in self.decode_layers:
            X = layer(X)
        return X

    def forward_pass(self, X):
        return self.decode(self.encode(X))

    def cost_function(self, X):
        prediction = self.forward_pass(X)
        cost = tf.reduce_mean((prediction - X) ** 2)
        return cost

    def train(self, X, optimizer):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            cost_value = self.cost_function(X)
        weight_grads = tape.gradient(cost_value, self.variables)

        optimizer.apply_gradients(zip(weight_grads, self.variables), global_step=tf.train.get_or_create_global_step())
        return

    def Save(self):
        self.save_weights(self.AE_checkpoint_path)

    def Load(self):
        self.load_weights(self.AE_checkpoint_path)
