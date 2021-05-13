import tensorflow as tf
from censai.definitions import DTYPE, lrelu4p
from .layers.conv_gru_component import ConvGRUBlock


class UnetModel(tf.keras.Model):
    def __init__(self, num_cell_features):
        super(UnetModel, self).__init__(dtype=DTYPE)
        numfeat_1, numfeat_2, numfeat_3, numfeat_4 = num_cell_features

        activation = lrelu4p
        self.STRIDE = 4
        self.Lc11 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3), activation=activation, padding='same')
        self.Lp13 = tf.keras.layers.Conv2D(numfeat_1 / 2, (7, 7), activation=activation, strides=self.STRIDE,
                                           padding='same')

        self.Lc21 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), activation=activation, padding='same')
        self.Lc22 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), activation=activation, padding='same')
        self.Lp23 = tf.keras.layers.Conv2D(numfeat_2 / 2, (7, 7), activation=activation, strides=self.STRIDE,
                                           padding='same')

        self.Lc31 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), activation=activation, padding='same')
        self.Lc32 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), activation=activation, padding='same')
        self.Lp33 = tf.keras.layers.Conv2D(numfeat_3 / 2, (7, 7), activation=activation, strides=self.STRIDE,
                                           padding='same')

        self.LcZ1 = tf.keras.layers.Conv2D(numfeat_4 / 2, (8, 8), activation=activation, padding='same')
        self.LcZ2 = tf.keras.layers.Conv2D(numfeat_4 / 2, (8, 8), activation=activation, padding='same')

        self.Lu61 = tf.keras.layers.Conv2DTranspose(numfeat_3 / 2, (6, 6), activation=activation, strides=self.STRIDE,
                                                    padding='same')
        self.Lc62 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), activation=activation, padding='same')
        self.Lc63 = tf.keras.layers.Conv2D(numfeat_3 / 2, (3, 3), activation=activation, padding='same')

        self.Lu71 = tf.keras.layers.Conv2DTranspose(numfeat_2 / 2, (6, 6), activation=activation, strides=self.STRIDE,
                                                    padding='same')
        self.Lc72 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), activation=activation, padding='same')
        self.Lc73 = tf.keras.layers.Conv2D(numfeat_2 / 2, (3, 3), activation=activation, padding='same')

        self.Lu81 = tf.keras.layers.Conv2DTranspose(numfeat_1 / 2, (6, 6), activation=activation, strides=self.STRIDE,
                                                    padding='same')
        self.Lc82 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3), activation=activation, padding='same')
        self.Lc83 = tf.keras.layers.Conv2D(numfeat_1 / 2, (3, 3), activation=activation, padding='same')

        self.Loutputs = tf.keras.layers.Conv2D(1, (2, 2), activation='linear', padding='same')

        self.GRU_COMP1 = ConvGRUBlock(numfeat_1)
        self.GRU_COMP2 = ConvGRUBlock(numfeat_2)
        self.GRU_COMP3 = ConvGRUBlock(numfeat_3)
        self.GRU_COMP4 = ConvGRUBlock(numfeat_4)

    def call(self, inputs, state, grad):
        stacked_input = tf.concat([inputs, grad], axis=3)
        ht_1, ht_2, ht_3, ht_4 = state

        c1 = self.Lc11(stacked_input)
        c1_gru, c1_gru_state = self.GRU_COMP1(c1, ht_1)

        p1 = self.Lp13(c1)
        c2 = self.Lc21(p1)
        c2 = self.Lc22(c2)
        c2_gru, c2_gru_state = self.GRU_COMP2(c2, ht_2)

        p2 = self.Lp23(c2)
        c3 = self.Lc31(p2)
        c3 = self.Lc32(c3)
        c3_gru, c3_gru_state = self.GRU_COMP3(c3, ht_3)

        p3 = self.Lp33(c3)

        z1 = self.LcZ1(p3)
        z1 = self.LcZ2(z1)
        c4_gru, c4_gru_state = self.GRU_COMP4(z1, ht_4)

        u6 = self.Lu61(c4_gru)
        u6 = tf.concat([u6, c3_gru], axis=3)
        c6 = self.Lc62(u6)
        c6 = self.Lc63(c6)

        u7 = self.Lu71(c6)
        u7 = tf.concat([u7, c2_gru], axis=3)
        c7 = self.Lc72(u7)
        c7 = self.Lc73(c7)

        u8 = self.Lu81(c7)
        u8 = tf.concat([u8, c1_gru], axis=3)
        c8 = self.Lc82(u8)
        c8 = self.Lc83(c8)

        delta_xt = self.Loutputs(c8)
        xt = inputs + delta_xt
        ht = [c1_gru_state, c2_gru_state, c3_gru_state, c4_gru_state]

        return xt, ht
