import tensorflow as tf
from censai.definitions import lrelu, m_softplus, DTYPE


class VAE(tf.keras.Model):
    def __init__(self, npix_side=256, channels=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        activation = lrelu
        self.npix_side = npix_side
        self.channels = 1
        n_downsample = 5
        self.l0 = tf.keras.layers.Conv2D(16, (4, 4), strides=2, padding='same', activation=activation)
        self.l1 = tf.keras.layers.Conv2D(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.l2 = tf.keras.layers.Conv2D(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.l3 = tf.keras.layers.Conv2D(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.l4 = tf.keras.layers.Conv2D(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.l5 = tf.keras.layers.Conv2D(16 * 1, (4, 4), strides=1, padding='same', activation=activation)
        self.l_mean = tf.keras.layers.Conv2D(16, (2, 2), strides=1, padding='same', activation=activation)
        self.l_logvar = tf.keras.layers.Conv2D(16, (2, 2), strides=1, padding='same', activation=activation)

        # inputs_decoder = 6**2 * 2
        # self.d1 = tf.keras.layers.Dense(inputs_decoder, activation=activation)
        # self.d2 = tf.keras.layers.Dense(inputs_decoder  , activation=activation)
        self.d3 = tf.keras.layers.Conv2DTranspose(16 * 4, (4, 4), strides=2, padding='same', activation=activation)
        self.d4 = tf.keras.layers.Conv2DTranspose(16 * 4, (4, 4), strides=2, padding='same', activation=activation)
        self.d5 = tf.keras.layers.Conv2DTranspose(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.d6 = tf.keras.layers.Conv2DTranspose(16 * 2, (4, 4), strides=2, padding='same', activation=activation)
        self.d7 = tf.keras.layers.Conv2DTranspose(16 * 1, (4, 4), strides=2, padding='same', activation=activation)
        self.d8 = tf.keras.layers.Conv2DTranspose(16, (4, 4), strides=1, padding='same', activation=activation)
        self.d9 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=1, padding='same', activation=m_softplus)

    def encoder(self, X_in):
        x = tf.reshape(X_in, shape=[-1, self.npix_side, self.npix_side, 1])
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        mean = self.l_mean(x)
        logvar = 0.5 * self.l_logvar(x)
        epsilon = tf.random.normal([x.shape[0], 8, 8, 16], dtype=DTYPE)
        z = mean + tf.multiply(epsilon, tf.exp(logvar))
        return z, mean, logvar

    def decoder(self, sampled_z):
        # reshaped_dim = [-1, 6, 6, 2]
        # x = self.d1(sampled_z)
        # x = self.d2(sampled_z)
        # x = tf.reshape(x, reshaped_dim)
        x = self.d3(sampled_z)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        img = tf.reshape(x, shape=[-1, self.npix_side, self.npix_side])
        return img

    def cost(self, X_in):
        sampled_code, mean, logvar = self.encoder(X_in)
        decoded_im = self.decoder(sampled_code)
        img_cost = tf.reduce_sum((decoded_im - X_in) ** 2, [1, 2])
        latent_cost = -0.5 * tf.reduce_sum(1.0 + 2.0 * logvar - tf.square(mean) - tf.exp(2.0 * logvar), axis=(1, 2, 3))
        cost = tf.reduce_mean(img_cost + latent_cost)
        return cost, decoded_im

    def draw_image(self, N):
        randoms = tf.random.normal((N, 8, 8, 16), dtype=DTYPE)
        simulated_im = self.decoder(randoms)
        return simulated_im