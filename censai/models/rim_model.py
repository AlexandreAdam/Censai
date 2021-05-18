import tensorflow as tf
from .layers.conv_gru import ConvGRU


class Model(tf.keras.Model):
    def __init__(
            self,
            num_cell_features,
            kernel_size=6,
            kernel_reg_amp=0.,
            bias_reg_amp=0.,
            initializer="glorot_normal",
            name="RIMModel"
    ):
        super(Model, self).__init__(name=name)
        self.num_gru_features = num_cell_features/2
        num_filt_emb1_1 = self.num_gru_features
        num_filt_emb1_2 = self.num_gru_features
        num_filt_emb2   = self.num_gru_features
        num_filt_emb3_1 = self.num_gru_features
        num_filt_emb3_2 = self.num_gru_features
        self.conv1_1 = tf.keras.layers.Conv2D(
                filters=num_filt_emb1_1,
                kernel_size=[kernel_size,kernel_size],
                strides=4,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer=initializer
                )
        self.conv1_2 = tf.keras.layers.Conv2D(
                filters = num_filt_emb1_2,
                kernel_size=[kernel_size,kernel_size],
                strides=4,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer = initializer
                )
        self.conv1_3 = tf.keras.layers.Conv2D(
                filters = num_filt_emb1_2,
                kernel_size=[kernel_size,kernel_size],
                strides=2,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer = initializer
                )
        self.conv2 = tf.keras.layers.Conv2D(
                filters=num_filt_emb2,
                kernel_size=[5,5],
                strides=1,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer=initializer
                )
        self.conv3_1 = tf.keras.layers.Conv2DTranspose(
                filters=num_filt_emb3_1,
                kernel_size=[kernel_size,kernel_size],
                strides=4,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer=initializer
                )
        self.conv3_2 = tf.keras.layers.Conv2DTranspose(
                filters=num_filt_emb3_2,
                kernel_size=[kernel_size,kernel_size],
                strides=4,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer=initializer
                )
        self.conv3_3 = tf.keras.layers.Conv2DTranspose(
                filters=num_filt_emb3_2,
                kernel_size=[kernel_size,kernel_size],
                strides=2,
                activation='relu',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer = initializer
                )
        self.conv4 = tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=[5, 5],
                strides=1,
                activation='linear',
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(l=kernel_reg_amp),
                bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp),
                kernel_initializer=initializer
                )
        self.gru1 = ConvGRU(self.num_gru_features)
        self.gru2 = ConvGRU(self.num_gru_features)

    def call(self, inputs, state, grad):
        stacked_input = tf.concat([inputs , grad], axis=3)
        xt_1E = self.conv1_1(stacked_input)
        xt_1E = self.conv1_2(xt_1E)
        xt_1E = self.conv1_3(xt_1E)
        ht_11 , ht_12 = tf.split(state, 2, axis=3)
        gru_1_out,_ = self.gru1(xt_1E ,ht_11)
        gru_1_outE = self.conv2(gru_1_out)
        gru_2_out,_ = self.gru2(gru_1_outE, ht_12)
        delta_xt_1 = self.conv3_1(gru_2_out)
        delta_xt_1 = self.conv3_2(delta_xt_1)
        delta_xt_1 = self.conv3_3(delta_xt_1)
        delta_xt = self.conv4(delta_xt_1)
        xt = delta_xt + inputs
        ht = tf.concat([gru_1_out, gru_2_out], axis=3)
        return xt, ht
