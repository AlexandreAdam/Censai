import tensorflow as tf
from censai.definitions import DTYPE, lrelu4p
from .layers.conv_gru_component import ConvGRUBlock


class UnetModel512(tf.keras.Model):
    def __init__(self, num_cell_features, strides=4):
        super(UnetModel512, self).__init__(dtype=DTYPE)
        numfeat_1, numfeat_2, numfeat_3, numfeat_4 = num_cell_features

        activation = lrelu4p
        self.strides = strides
        common_params = {"activation": activation, "padding": "same"}
        self.Lc11 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3),  **common_params)
        self.Lp13 = tf.keras.layers.Conv2D(numfeat_1 / 2, (7, 7), strides=strides, **common_params)

        self.Lc21 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), **common_params)
        self.Lc22 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), **common_params)
        self.Lp23 = tf.keras.layers.Conv2D(numfeat_2 / 2, (7, 7), strides=strides, **common_params)

        self.Lc31 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), **common_params)
        self.Lc32 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), **common_params)
        self.Lp33 = tf.keras.layers.Conv2D(numfeat_3 / 2, (7, 7), strides=strides, **common_params)

        self.LcZ1 = tf.keras.layers.Conv2D(numfeat_4 / 2, (8, 8), **common_params)
        self.LcZ2 = tf.keras.layers.Conv2D(numfeat_4 / 2, (8, 8), **common_params)

        self.Lu61 = tf.keras.layers.Conv2DTranspose(numfeat_3 / 2, (6, 6), strides=strides, **common_params)
        self.Lc62 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), **common_params)
        self.Lc63 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), **common_params)

        self.Lu71 = tf.keras.layers.Conv2DTranspose(numfeat_2 / 2, (6, 6), strides=strides, **common_params)
        self.Lc72 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), **common_params)
        self.Lc73 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), **common_params)

        self.Lu81 = tf.keras.layers.Conv2DTranspose(numfeat_1 / 2, (6, 6), strides=strides, **common_params)
        self.Lc82 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3), **common_params)
        self.Lc83 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3), **common_params)

        self.Loutputs = tf.keras.layers.Conv2D(1, (2, 2), activation='linear', padding='same')

        self.GRU_COMP1 = ConvGRUBlock(numfeat_1)
        self.GRU_COMP2 = ConvGRUBlock(numfeat_2)
        self.GRU_COMP3 = ConvGRUBlock(numfeat_3)
        self.GRU_COMP4 = ConvGRUBlock(numfeat_4)

    def call(self, inputs, state, grad):
        x = tf.concat([inputs, grad], axis=3)  # x is short for Delta x_t
        ht_1, ht_2, ht_3, ht_4 = state

        x = self.Lc11(x)
        c1_gru, c1_gru_state = self.GRU_COMP1(x, ht_1)  # GRU in the middle of the skip connection
        x = self.Lp13(x)  # downsampling layer

        x = self.Lc21(x)
        x = self.Lc22(x)
        c2_gru, c2_gru_state = self.GRU_COMP2(x, ht_2)
        x = self.Lp23(x)  # downsampling layer

        x = self.Lc31(x)
        x = self.Lc32(x)
        c3_gru, c3_gru_state = self.GRU_COMP3(x, ht_3)
        x = self.Lp33(x)  # downsampling layer

        x = self.LcZ1(x)  # bottleneck of the architecture
        x = self.LcZ2(x)
        x, c4_gru_state = self.GRU_COMP4(x, ht_4)

        x = self.Lu61(x)   # upsampling layer
        x = tf.concat([x, c3_gru], axis=3)  # skip connection
        x = self.Lc62(x)
        x = self.Lc63(x)

        x = self.Lu71(x)  # upsampling layer
        x = tf.concat([x, c2_gru], axis=3)  # skip connection
        x = self.Lc72(x)
        x = self.Lc73(x)

        x = self.Lu81(x)  # upsampling layer
        x = tf.concat([x, c1_gru], axis=3)  # skip connection
        x = self.Lc82(x)
        x = self.Lc83(x)

        delta_xt = self.Loutputs(x)
        xt = inputs + delta_xt  # update image
        ht = [c1_gru_state, c2_gru_state, c3_gru_state, c4_gru_state]

        return xt, ht
