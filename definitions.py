from tensorflow.python.keras.layers.merge import concatenate
from astropy.cosmology import Planck15 as cosmo

def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))

def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus( -x -5.0 ) 

def xsquared(x):
    return (x/4)**2


datatype = tf.float32
class VAE(tf.keras.Model):
    def __init__(self , npix_side = 256):
        super(VAE, self).__init__()
        activation = lrelu
        self.npix_side = npix_side
        # self.n_latent = n_latent
        n_downsample = 5
        self.l0 = tf.keras.layers.Conv2D(16, (4,4), strides=2, padding='same', activation=activation)
        self.l1 = tf.keras.layers.Conv2D(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.l2 = tf.keras.layers.Conv2D(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.l3 = tf.keras.layers.Conv2D(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.l4 = tf.keras.layers.Conv2D(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.l5 = tf.keras.layers.Conv2D(16*1, (4,4), strides=1, padding='same', activation=activation)
        self.l_mn = tf.keras.layers.Conv2D(16, (2,2), strides=1, padding='same', activation=activation)
        self.l_sd = tf.keras.layers.Conv2D(16, (2,2), strides=1, padding='same', activation=activation)
        # self.l6 = tf.keras.layers.Dense(n_latent)
        # self.l7 = tf.keras.layers.Dense(n_latent)
        
        # inputs_decoder = 6**2 * 2
        #self.d1 = tf.keras.layers.Dense(inputs_decoder, activation=activation)
        # self.d2 = tf.keras.layers.Dense(inputs_decoder  , activation=activation)
        self.d3 = tf.keras.layers.Conv2DTranspose(16*4, (4,4), strides=2, padding='same', activation=activation)
        self.d4 = tf.keras.layers.Conv2DTranspose(16*4, (4,4), strides=2, padding='same', activation=activation)
        self.d5 = tf.keras.layers.Conv2DTranspose(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.d6 = tf.keras.layers.Conv2DTranspose(16*2, (4,4), strides=2, padding='same', activation=activation)
        self.d7 = tf.keras.layers.Conv2DTranspose(16*1, (4,4), strides=2, padding='same', activation=activation)
        self.d8 = tf.keras.layers.Conv2DTranspose(16, (4,4), strides=1, padding='same', activation=activation)
        self.d9 = tf.keras.layers.Conv2DTranspose(1, (4,4), strides=1, padding='same', activation=m_softplus)
    def encoder(self, X_in):
        x = tf.reshape(X_in, shape=[-1, self.npix_side, self.npix_side, 1])
        x = self.l0(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        # x = tf.contrib.layers.flatten(x)
        mn = self.l_mn(x)
        sd       = 0.5 * self.l_sd(x)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], 8 ,8 , 16]) , dtype=datatype ) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))        
        return z, mn, sd

    def decoder(self, sampled_z):            
        # reshaped_dim = [-1, 6, 6, 2]
        #x = self.d1(sampled_z)
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
        sampled_code, mn, sd = self.encoder(X_in)
        decoded_im = self.decoder(sampled_code)
        img_cost = tf.reduce_sum( (decoded_im - X_in)**2 , [1,2] )
        latent_cost = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), (1,2,3))
        cost = tf.reduce_mean(img_cost + latent_cost)
        return cost , decoded_im
    def draw_image(self, N):
        randoms = tf.random_normal(  (N , 8,8,16) , dtype=datatype)
        simulated_im = self.decoder(randoms)
	return simulated_im
        



class RayTracer(tf.keras.Model):
    def __init__(self,trainable=True):
        super(RayTracer, self).__init__()

        self.Lc11 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same',trainable=trainable) 
        self.Lc12 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same') 
        self.Lp13 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', strides=(2, 2), padding='same') 

        self.Lc21 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc22 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lp23 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', strides=(2, 2), padding='same')

        self.Lc31 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc32 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lp33 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', strides=(2, 2), padding='same')

        self.Lc41 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc42 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lp43 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', strides=(2, 2), padding='same')

        self.Lc51 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc52 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lp53 = tf.keras.layers.Conv2D(32, (6, 6), activation='linear', strides=(4, 4), padding='same')

        self.LcZ1 = tf.keras.layers.Conv2D(32, (16, 16), activation='linear', padding='same')
        self.LcZ2 = tf.keras.layers.Conv2D(32, (16, 16), activation='linear', padding='same')

        self.Lu61 = tf.keras.layers.Conv2DTranspose(32, (6, 6), activation='linear', strides=(4, 4), padding='same')
        self.Lc62 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc63 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')

        self.Lu71 = tf.keras.layers.Conv2DTranspose(32, (2, 2), activation='linear', strides=(2, 2), padding='same')
        self.Lc72 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc73 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')

        self.Lu81 = tf.keras.layers.Conv2DTranspose(32, (2, 2), activation='linear', strides=(2, 2), padding='same')
        self.Lc82 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc83 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')

        self.Lu91 = tf.keras.layers.Conv2DTranspose(32, (2, 2), activation='linear', strides=(2, 2), padding='same')
        self.Lc92 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc93 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')

        self.Lu101 = tf.keras.layers.Conv2DTranspose(32, (2, 2), activation='linear', strides=(2, 2), padding='same')
        self.Lc102 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')
        self.Lc103 = tf.keras.layers.Conv2D(32, (3, 3), activation='linear', padding='same')

        self.Loutputs = tf.keras.layers.Conv2D(2, (1, 1), activation='linear')

    def call(self, kappa):
        c1 = self.Lc11 (kappa)
        c1 = self.Lc12 (c1)
        p1 = self.Lp13 (c1)

        c2 = self.Lc21 (p1)
        c2 = self.Lc22 (c2)
        p2 = self.Lp23 (c2)

        c3 = self.Lc31 (p2)
        c3 = self.Lc32 (c3)
        p3 = self.Lp33 (c3)

        c4 = self.Lc41 (p3)
        c4 = self.Lc42 (c4)
        p4 = self.Lp43 (c4)

        c5 = self.Lc51 (p4)
        c5 = self.Lc52 (c5)
        p5 = self.Lp53 (c5)

        z1 = self.LcZ1 (p5)
        z1 = self.LcZ2 (z1)

        u6 = self.Lu61 (z1)
        u6 = tf.concat([u6, c5], axis=3)
        c6 = self.Lc62 (u6)
        c6 = self.Lc63 (c6)

        u7 = self.Lu71 (c6)
        u7 = tf.concat([u7, c4], axis=3)
        c7 = self.Lc72 (u7)
        c7 = self.Lc73 (c7)

        u8 = self.Lu81 (c7)
        u8 = tf.concat([u8, c3], axis=3)
        c8 = self.Lc82 (u8)
        c8 = self.Lc83 (c8)

        u9 = self.Lu91 (c8)
        u9 = tf.concat([u9, c2], axis=3)
        c9 = self.Lc92 (u9)
        c9 = self.Lc93 (c9)

        u10 = self.Lu101 (c9)
        u10 = tf.concat([u10, c1], axis=3)
        c10 = self.Lc102 (u10)
        c10 = self.Lc103 (c10)

        outputs = self.Loutputs (c10)
        
        return outputs

    def cost_function( self, kappa , x_a_label , y_a_label):
        alpha = self.call(kappa)
        alpha_label = tf.concat([x_a_label , y_a_label] , axis=3)
        return tf.reduce_mean( ( alpha - alpha_label)**2 ) , alpha


# class RayTracer(tf.keras.Model):
#     def __init__(self):
#         super(RayTracer, self).__init__()
#
#         self.conv_0 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[4,4], strides=1, activation='linear',padding='same')
#
#         self.conv_ds_1 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_ds_2 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_ds_3 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_ds_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#
#         self.conv_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size=[12,12], strides=1, activation='linear',padding='same')
#
#         self.conv_us_1 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#
#         self.conv_us_21 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_22 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#
#         self.conv_us_31 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_32 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_33 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#
#         self.conv_us_41 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_42 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_43 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#         self.conv_us_44 = tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size=[4,4], strides=2, activation='linear',padding='same')
#
#         self.conv_merge = tf.keras.layers.Conv2D(filters = 2, kernel_size=[4,4], strides=1, activation='linear',padding='same')
#
#
#     def call(self, kappa):
#         x00 = self.conv_0(kappa)
#         xds1 = self.conv_ds_1(kappa)
#         xds2 = self.conv_ds_2(xds1)
#         xds3 = self.conv_ds_3(xds2)
#         xds4 = self.conv_ds_4(xds3)
#
#         xds4 = self.conv_4(xds4)
#
#         xus11 = self.conv_us_1(xds1)
#
#         xus21 = self.conv_us_21(xds2)
#         xus22 = self.conv_us_22(xus21)
#
#         xus31 = self.conv_us_31(xds3)
#         xus32 = self.conv_us_32(xus31)
#         xus33 = self.conv_us_33(xus32)
#
#         xus41 = self.conv_us_41(xds4)
#         xus42 = self.conv_us_42(xus41)
#         xus43 = self.conv_us_43(xus42)
#         xus44 = self.conv_us_44(xus43)
#
#
#         stacked_input = tf.concat([x00 , xus11 , xus22 , xus33 , xus44], axis=3)
#         x_a = self.conv_merge(stacked_input)
#
#         return x_a
#     def cost_function( self, kappa , x_a_label , y_a_label):
#         alpha = self.call(kappa)
#     alpha_label = tf.concat([x_a_label , y_a_label] , axis=3)
#         return tf.reduce_mean( ( alpha - x_a_label)**2 )


kernel_size = 5

class Conv_GRU(tf.keras.Model):
    def __init__(self , num_features):
        super(Conv_GRU, self).__init__()
        num_filters = num_features
        self.conv_1 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='sigmoid',padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='sigmoid',padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters = num_filters, kernel_size=[kernel_size,kernel_size], strides=1, activation='tanh',padding='same')
    def call(self, inputs, state):
        stacked_input = tf.concat([inputs , state], axis=3)
        z = self.conv_1(stacked_input)
        r = self.conv_2(stacked_input)
        r_state = tf.multiply(r , state)
        stacked_r_state = tf.concat([inputs , r_state], axis=3)
        update_info = self.conv_3(stacked_r_state)
        new_state = tf.multiply( 1-z , state) + tf.multiply(z , update_info)
        return new_state , new_state

initializer = tf.initializers.random_normal( stddev=0.06)
kernal_reg_amp = 0.0
bias_reg_amp = 0.0
kernel_size = 6

class Model(tf.keras.Model):
    def __init__(self,num_cell_features):
        super(Model, self).__init__()
        self.num_gru_features = num_cell_features/2
        num_filt_emb1_1 = self.num_gru_features
        num_filt_emb1_2 = self.num_gru_features
        num_filt_emb2 = self.num_gru_features
        num_filt_emb3_1 = self.num_gru_features
        num_filt_emb3_2 = self.num_gru_features
        self.conv1_1 = tf.keras.layers.Conv2D(filters = num_filt_emb1_1, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv1_2 = tf.keras.layers.Conv2D(filters = num_filt_emb1_2, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv1_3 = tf.keras.layers.Conv2D(filters = num_filt_emb1_2, kernel_size=[kernel_size,kernel_size], strides=2, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters = num_filt_emb2, kernel_size=[5,5], strides=1, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_1 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_1, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_2 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_2, kernel_size=[kernel_size,kernel_size], strides=4, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv3_3 = tf.keras.layers.Conv2DTranspose(filters = num_filt_emb3_2, kernel_size=[kernel_size,kernel_size], strides=2, activation='relu',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.conv4 = tf.keras.layers.Conv2D(filters = 1, kernel_size=[5,5], strides=1, activation='linear',padding='same',kernel_regularizer= tf.keras.regularizers.l2(l=kernal_reg_amp), bias_regularizer=tf.keras.regularizers.l2(l=bias_reg_amp), kernel_initializer = initializer)
        self.gru1 = Conv_GRU(self.num_gru_features)
        self.gru2 = Conv_GRU(self.num_gru_features)
    def call(self, inputs, state, grad):
        stacked_input = tf.concat([inputs , grad], axis=3)
        xt_1E = self.conv1_1(stacked_input)
        xt_1E = self.conv1_2(xt_1E)
        xt_1E = self.conv1_3(xt_1E)
        ht_11 , ht_12 = tf.split(state, 2, axis=3)
        gru_1_out,_ = self.gru1( xt_1E ,ht_11)
        gru_1_outE = self.conv2(gru_1_out)
        gru_2_out,_ = self.gru2( gru_1_outE ,ht_12)
        delta_xt_1 = self.conv3_1(gru_2_out)
        delta_xt_1 = self.conv3_2(delta_xt_1)
        delta_xt_1 = self.conv3_3(delta_xt_1)
        delta_xt = self.conv4(delta_xt_1)
        xt = delta_xt + inputs
        ht = tf.concat([gru_1_out , gru_2_out], axis=3)
        return xt, ht

    
    
class GRU_COMPONENT(tf.keras.Model):
    def __init__(self,num_cell_features):
        super(GRU_COMPONENT, self).__init__()
        self.kernel_size = 5
        self.num_gru_features = num_cell_features/2
        self.conv1 = tf.keras.layers.Conv2D(filters = self.num_gru_features, kernel_size=self.kernel_size, strides=1, activation='relu',padding='same')
        self.gru1 = Conv_GRU(self.num_gru_features)
        self.gru2 = Conv_GRU(self.num_gru_features)
    def call(self, inputs, state):
        ht_11 , ht_12 = tf.split(state, 2, axis=3)
        gru_1_out,_ = self.gru1( inputs ,ht_11)
        gru_1_outE = self.conv1(gru_1_out)
        gru_2_out,_ = self.gru2( gru_1_outE ,ht_12)
        ht = tf.concat([gru_1_out , gru_2_out], axis=3)
        xt = gru_2_out
        return xt, ht

def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))

class RIM_UNET(tf.keras.Model):
    def __init__(self,num_cell_features):
        super(RIM_UNET, self).__init__()
        numfeat_1 , numfeat_2 , numfeat_3 , numfeat_4 = num_cell_features
        
        activation = lrelu4p
        self.STRIDE = 4
        self.Lc11 = tf.keras.layers.Conv2D(numfeat_1/2, (3, 3), activation=activation, padding='same') 
        self.Lp13 = tf.keras.layers.Conv2D(numfeat_1/2, (7, 7), activation=activation, strides=self.STRIDE, padding='same') 

        self.Lc21 = tf.keras.layers.Conv2D(numfeat_2/2, (3, 3), activation=activation, padding='same')
        self.Lc22 = tf.keras.layers.Conv2D(numfeat_2/2, (3, 3), activation=activation, padding='same')
        self.Lp23 = tf.keras.layers.Conv2D(numfeat_2/2, (7, 7), activation=activation, strides=self.STRIDE, padding='same')

        self.Lc31 = tf.keras.layers.Conv2D(numfeat_3/2, (3, 3), activation=activation, padding='same')
        self.Lc32 = tf.keras.layers.Conv2D(numfeat_3/2, (3, 3), activation=activation, padding='same')
        self.Lp33 = tf.keras.layers.Conv2D(numfeat_3/2, (7, 7), activation=activation, strides=self.STRIDE, padding='same')

        self.LcZ1 = tf.keras.layers.Conv2D(numfeat_4/2, (8, 8), activation=activation, padding='same')
        self.LcZ2 = tf.keras.layers.Conv2D(numfeat_4/2, (8, 8), activation=activation, padding='same')

        self.Lu61 = tf.keras.layers.Conv2DTranspose(numfeat_3/2, (6, 6), activation=activation, strides=self.STRIDE, padding='same')
        self.Lc62 = tf.keras.layers.Conv2D(numfeat_3/2, (3, 3), activation=activation, padding='same')
        self.Lc63 = tf.keras.layers.Conv2D(numfeat_3/2, (3, 3), activation=activation, padding='same')

        self.Lu71 = tf.keras.layers.Conv2DTranspose(numfeat_2/2, (6, 6), activation=activation, strides=self.STRIDE, padding='same')
        self.Lc72 = tf.keras.layers.Conv2D(numfeat_2/2, (3, 3), activation=activation, padding='same')
        self.Lc73 = tf.keras.layers.Conv2D(numfeat_2/2, (3, 3), activation=activation, padding='same')

        self.Lu81 = tf.keras.layers.Conv2DTranspose(numfeat_1/2, (6, 6), activation=activation, strides=self.STRIDE, padding='same')
        self.Lc82 = tf.keras.layers.Conv2D(numfeat_1/2, (3, 3), activation=activation, padding='same')
        self.Lc83 = tf.keras.layers.Conv2D(numfeat_1/2, (3, 3), activation=activation, padding='same')

        self.Loutputs = tf.keras.layers.Conv2D(1, (2, 2), activation='linear', padding='same')

        self.GRU_COMP1 = GRU_COMPONENT(numfeat_1)
        self.GRU_COMP2 = GRU_COMPONENT(numfeat_2)
        self.GRU_COMP3 = GRU_COMPONENT(numfeat_3)
        self.GRU_COMP4 = GRU_COMPONENT(numfeat_4)
        
    def call(self, inputs, state, grad):
        
        stacked_input = tf.concat([inputs , grad], axis=3)
        ht_1 , ht_2 , ht_3 , ht_4 = state
        
        c1 = self.Lc11 (stacked_input)
        c1_gru , c1_gru_state = self.GRU_COMP1(c1 , ht_1)
        
        p1 = self.Lp13 (c1)
        c2 = self.Lc21 (p1)
        c2 = self.Lc22 (c2)
        c2_gru , c2_gru_state = self.GRU_COMP2(c2 , ht_2)
        
        p2 = self.Lp23 (c2)
        c3 = self.Lc31 (p2)
        c3 = self.Lc32 (c3)
        c3_gru , c3_gru_state = self.GRU_COMP3(c3 , ht_3)

        p3 = self.Lp33(c3)
        
        z1 = self.LcZ1 (p3)
        z1 = self.LcZ2 (z1)
        c4_gru , c4_gru_state = self.GRU_COMP4(z1 , ht_4)

        u6 = self.Lu61 (c4_gru)
        u6 = tf.concat([u6, c3_gru], axis=3)
        c6 = self.Lc62 (u6)
        c6 = self.Lc63 (c6)

        u7 = self.Lu71 (c6)
        u7 = tf.concat([u7, c2_gru], axis=3)
        c7 = self.Lc72 (u7)
        c7 = self.Lc73 (c7)

        u8 = self.Lu81 (c7)
        u8 = tf.concat([u8, c1_gru], axis=3)
        c8 = self.Lc82 (u8)
        c8 = self.Lc83 (c8)

        delta_xt = self.Loutputs(c8)
        xt = inputs + delta_xt
        ht = [c1_gru_state , c2_gru_state , c3_gru_state , c4_gru_state]

        return xt, ht



class RIM_UNET_CELL(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, num_steps ,num_pixels, state_size , input_size=None, activation=tf.tanh,cond1= None, cond2= None):
        self.num_pixels = num_pixels
        self.num_steps = num_steps
        self._num_units = state_size
        self.double_RIM_state_size = state_size
        self.single_RIM_state_size = state_size/2
        self.gru_state_size = state_size/4
        self.gru_state_pixel_downsampled = 16*2
        self._activation = activation
        self.state_size_list = [8 , 16 , 32 , 64]
        self.model_1 = RIM_UNET(self.state_size_list)
        self.model_2 = RIM_UNET(self.state_size_list)
        self.batch_size = batch_size
        self.initial_condition(cond1, cond2)
        self.initial_output_state()

    def initial_condition(self, cond1, cond2):
        if cond1 is None:
            self.inputs_1 = tf.zeros(shape=(self.batch_size , self.num_pixels , self.num_pixels , 1))
        else:
            self.inputs_1 = tf.identity(cond1)
            
        if cond2 is None:
            self.inputs_2 = tf.zeros(shape=(self.batch_size , self.num_pixels , self.num_pixels , 1))
        else:
            self.inputs_2 = tf.identity(cond2)
            
        return
            
        
    def initial_output_state(self):

        STRIDE = self.model_1.STRIDE
        numfeat_1 , numfeat_2 , numfeat_3 , numfeat_4 = self.state_size_list
        state_11 = tf.zeros(shape=(self.batch_size,  self.num_pixels, self.num_pixels , numfeat_1 ))
        state_12 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**1, self.num_pixels/STRIDE**1 , numfeat_2 ))
        state_13 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**2, self.num_pixels/STRIDE**2 , numfeat_3 ))
        state_14 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**3, self.num_pixels/STRIDE**3 , numfeat_4 ))
        self.state_1 = [state_11 , state_12 , state_13 , state_14]

        STRIDE = self.model_2.STRIDE
        state_21 = tf.zeros(shape=(self.batch_size,  self.num_pixels, self.num_pixels , numfeat_1 ))
        state_22 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**1, self.num_pixels/STRIDE**1 , numfeat_2 ))
        state_23 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**2, self.num_pixels/STRIDE**2 , numfeat_3 ))
        state_24 = tf.zeros(shape=(self.batch_size,  self.num_pixels/STRIDE**3, self.num_pixels/STRIDE**3 , numfeat_4 ))
        self.state_2 = [state_21 , state_22 , state_23 , state_24]

        


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2 , scope=None):
        xt_1, ht_1 = self.model_1(inputs_1, state_1 , grad_1)
        xt_2, ht_2 = self.model_2(inputs_2, state_2 , grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def forward_pass(self, data):

        output_series_1 = []
        output_series_2 = []

        with tf.GradientTape() as g:
            g.watch(self.inputs_1)
            g.watch(self.inputs_2)
            y = log_likelihood(data,physical_model(self.inputs_1,self.inputs_2),noise_rms)
        grads = g.gradient(y, [self.inputs_1 , self.inputs_2])

        output_1, state_1, output_2, state_2 = self.__call__(self.inputs_1, self.state_1 , grads[0] , self.inputs_2 , self.state_2 , grads[1])
        output_series_1.append(output_1)
        output_series_2.append(output_2)

        for current_step in range(self.num_steps-1):
            with tf.GradientTape() as g:
                g.watch(output_1)
                g.watch(output_2)
                y = log_likelihood(data,physical_model(output_1,output_2),noise_rms)
            grads = g.gradient(y, [output_1 , output_2])


            output_1, state_1 , output_2 , state_2 = self.__call__(output_1, state_1 , grads[0] , output_2 , state_2 , grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)
        final_log_L = log_likelihood(data,physical_model(output_1,output_2),noise_rms)
        return output_series_1 , output_series_2 , final_log_L

    def cost_function( self, data,labels_x_1,labels_x_2):
        output_series_1 , output_series_2 , final_log_L = self.forward_pass(data)
        return tf.reduce_mean(tf.square(output_series_1 - labels_x_1)) + tf.reduce_mean(tf.square(output_series_2 - labels_x_2)), output_series_1 , output_series_2 , output_series_1[-1].numpy() , output_series_2[-1].numpy()

    
    
class RIM_CELL(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, num_steps ,num_pixels, state_size , input_size=None, activation=tf.tanh):
	self.num_pixels = num_pixels
        self.num_steps = num_steps
        self._num_units = state_size
        self.double_RIM_state_size = state_size
        self.single_RIM_state_size = state_size/2
        self.gru_state_size = state_size/4
	self.gru_state_pixel_downsampled = 16*2
        self._activation = activation
        self.model_1 = Model(self.single_RIM_state_size)
        self.model_2 = Model(self.single_RIM_state_size)
	self.batch_size = batch_size
	self.initial_output_state()

    def initial_output_state(self):
        self.inputs_1 = tf.zeros(shape=(self.batch_size , self.num_pixels , self.num_pixels , 1))
        self.state_1 = tf.zeros(shape=(self.batch_size,  self.num_pixels/self.gru_state_pixel_downsampled, self.num_pixels/self.gru_state_pixel_downsampled , self.single_RIM_state_size ))
        self.inputs_2 = tf.zeros(shape=(self.batch_size , self.num_pixels , self.num_pixels , 1))
        self.state_2 = tf.zeros(shape=(self.batch_size,  self.num_pixels/self.gru_state_pixel_downsampled, self.num_pixels/self.gru_state_pixel_downsampled , self.single_RIM_state_size ))


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs_1, state_1, grad_1, inputs_2, state_2, grad_2 , scope=None):
        xt_1, ht_1 = self.model_1(inputs_1, state_1 , grad_1)
        xt_2, ht_2 = self.model_2(inputs_2, state_2 , grad_2)
        return xt_1, ht_1, xt_2, ht_2

    def forward_pass(self, data):

	if (data.shape[0] != self.batch_size):
	    self.batch_size = data.shape[0]
	    self.initial_output_state()


        output_series_1 = []
        output_series_2 = []

        with tf.GradientTape() as g:
            g.watch(self.inputs_1)
            g.watch(self.inputs_2)
            y = log_likelihood(data,physical_model(self.inputs_1,self.inputs_2),noise_rms)
        grads = g.gradient(y, [self.inputs_1 , self.inputs_2])

        output_1, state_1, output_2, state_2 = self.__call__(self.inputs_1, self.state_1 , grads[0] , self.inputs_2 , self.state_2 , grads[1])
        output_series_1.append(output_1)
        output_series_2.append(output_2)

        for current_step in range(self.num_steps-1):
            with tf.GradientTape() as g:
                g.watch(output_1)
                g.watch(output_2)
                y = log_likelihood(data,physical_model(output_1,output_2),noise_rms)
            grads = g.gradient(y, [output_1 , output_2])


            output_1, state_1 , output_2 , state_2 = self.__call__(output_1, state_1 , grads[0] , output_2 , state_2 , grads[1])
            output_series_1.append(output_1)
            output_series_2.append(output_2)
        final_log_L = log_likelihood(data,physical_model(output_1,output_2),noise_rms)
        return output_series_1 , output_series_2 , final_log_L

    def cost_function( self, data,labels_x_1,labels_x_2):
        output_series_1 , output_series_2 , final_log_L = self.forward_pass(data)
        return tf.reduce_mean(tf.square(output_series_1 - labels_x_1)) + tf.reduce_mean(tf.square(output_series_2 - labels_x_2)), output_series_1 , output_series_2 , output_series_1[-1].numpy() , output_series_2[-1].numpy()

class SRC_KAPPA_Generator(object):

    def __init__(self,train_batch_size=1,test_batch_size=1,kap_side_length=7.68,src_side=3.0,num_src_side=48,num_kappa_side=48):
        self.src_side = src_side
        self.kap_side_length = kap_side_length
        self.num_src_side = num_src_side
        self.num_kappa_side = num_kappa_side
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        x_s = np.linspace(-1., 1., self.num_src_side,dtype='float32') * self.src_side/2
        y_s = np.linspace(-1., 1., self.num_src_side,dtype='float32') * self.src_side/2
        self.dy_s = y_s[1] - y_s[0]
        self.Xsrc, self.Ysrc = np.meshgrid(x_s, y_s)
        x_k = np.linspace(-1., 1., self.num_kappa_side,dtype='float32') * self.kap_side_length/2
        y_k = np.linspace(-1., 1., self.num_kappa_side,dtype='float32') * self.kap_side_length/2
        self.dy_k = y_k[1] - y_k[0]
        self.Xkap, self.Ykap = np.meshgrid(x_k, y_k)
        self.Kappa_tr = np.zeros((train_batch_size,num_kappa_side,num_kappa_side,1),dtype='float32')
        self.Source_tr = np.zeros((train_batch_size,num_src_side,num_src_side,1),dtype='float32')
        self.Kappa_ts = np.zeros((test_batch_size,num_kappa_side,num_kappa_side,1),dtype='float32')
        self.Source_ts = np.zeros((test_batch_size,num_src_side,num_src_side,1),dtype='float32')

    def Kappa_fun(self, xlens, ylens, elp, phi, Rein, rc=0,  Zlens = 0.5 , Zsource = 2.0, c = 299800000):
        Dds= cosmo.angular_diameter_distance_z1z2(Zlens,Zsource).value  * 1e6
        Ds = cosmo.angular_diameter_distance(Zsource).value  * 1e6
        sigma_v = np.sqrt( c**2/(4*np.pi)*Rein*np.pi/180/3600  * Ds/Dds )
        A = self.dy_k/2. *(2*np.pi/ (360*3600) )
        rcord, thetacord = np.sqrt(self.Xkap**2 + self.Ykap**2) , np.arctan2(self.Ykap, self.Xkap)
        thetacord = thetacord - phi
        Xkap, Ykap = rcord*np.cos(thetacord), rcord*np.sin(thetacord)
        rlens, thetalens = np.sqrt(xlens**2 + ylens**2) , np.arctan2(ylens, xlens)
        thetalens = thetalens - phi
        xlens, ylens = rlens*np.cos(thetalens), rlens*np.sin(thetalens)
        r = np.sqrt((Xkap-xlens)**2 + ((Ykap-ylens) * (1-elp) )**2) *(2*np.pi/ (360*3600) )
        Rein = (4*np.pi*sigma_v**2/c**2) * Dds /Ds
        kappa = np.divide( np.sqrt(1-elp)* Rein ,  (2* np.sqrt( r**2 + rc**2)))
        mass_inside_00_pix = 2.*A*(np.log(2.**(1./2.) + 1.) - np.log(2.**(1./2.)*A - A) + np.log(3.*A + 2.*2.**(1./2.)*A))
        density_00_pix = np.sqrt(1.-elp) * Rein/(2.) * mass_inside_00_pix/((2.*A)**2.)
        ind = np.argmin(r)
        kappa.flat[ind] = density_00_pix
        return kappa


    def gen_source(self, x_src = 0, y_src = 0, sigma_src = 1, norm = False):
        Im = np.exp( -(((self.Xsrc-x_src)**2 +(self.Ysrc-y_src)**2) / (2.*sigma_src**2) ))
        if norm is True:
            Im = Im/np.max(Im)
        return Im

    def draw_k_s(self,train_or_test):
        if (train_or_test=="train"):
            np.random.seed(seed=None)
            num_samples = self.train_batch_size
        if (train_or_test=="test"):
            np.random.seed(seed=136)
            num_samples = self.test_batch_size
        
        
        for i in range(num_samples):
            #parameters for kappa
            #np.random.seed(seed=155)
            xlens = np.random.uniform(low=-1.0, high=1.)
            ylens = np.random.uniform(low=-1.0, high=1.)
            elp = np.random.uniform(low=0.01, high=0.6)
            phi = np.random.uniform(low=0.0, high=2.*np.pi)
            Rein = np.random.uniform(low=2.0, high = 4.0)

            #parameters for source
            sigma_src = np.random.uniform(low=0.5, high=1.0)
            x_src = np.random.uniform(low=-0.5, high=0.5)
            y_src = np.random.uniform(low=-0.5, high=0.5)
            norm_source = True
            
            if (train_or_test=="train"):
                self.Source_tr[i,:,:,0] = self.gen_source(x_src = x_src, y_src = y_src, sigma_src = sigma_src, norm = norm_source)
                self.Kappa_tr[i,:,:,0]  = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
            if (train_or_test=="test"):
                self.Source_ts[i,:,:,0] = self.gen_source(x_src = x_src, y_src = y_src, sigma_src = sigma_src, norm = norm_source)
                self.Kappa_ts[i,:,:,0]  = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
        return
    def draw_average_k_s(self):
        src = self.gen_source(x_src = 0., y_src = 0., sigma_src = 0.5, norm = True)
        kappa = self.Kappa_fun( 0., 0. , 0.02, 0., 3.0)
        src = src.reshape(1, self.num_src_side,self.num_src_side, 1)
        kappa = kappa.reshape(1, self.num_kappa_side , self.num_kappa_side , 1)
        
        src=np.repeat(src,self.train_batch_size,axis=0)
        kappa=np.repeat(kappa,self.train_batch_size,axis=0)
        return src, kappa



class lens_util(object):

    def __init__(self, im_side= 7.68, src_side=3.0, numpix_side = 192 , kap_side=7.68 , method = "conv2d"):
        self.im_side = im_side
        self.numpix_side = numpix_side
        self.src_side     = src_side
        self.kap_numpix = numpix_side
        self.RT = RayTracer(trainable=False)
        checkpoint_path = "checkpoints/model_Unet_test"
        self.RT.load_weights(checkpoint_path)
        self.method = method
        self.set_deflection_angles_vars(kap_side)
                
    def set_deflection_angles_vars(self,kap_side):
        self.kap_side = kap_side
        self.kernel_side_l = self.kap_numpix*2+1;
        self.cond = np.zeros((self.kernel_side_l,self.kernel_side_l))
        self.cond[self.kap_numpix,self.kap_numpix] = True
        self.dx_kap = self.kap_side/(self.kap_numpix-1)
        x = tf.linspace(-1., 1., self.kernel_side_l)*self.kap_side
        y = tf.linspace(-1., 1., self.kernel_side_l)*self.kap_side
        X_filt, Y_filt = tf.meshgrid(x, y)
        kernel_denom = tf.square(X_filt) + tf.square(Y_filt)
        Xconv_kernel = tf.divide(-X_filt , kernel_denom)
        B = tf.zeros_like(Xconv_kernel)
        Xconv_kernel = tf.where(self.cond,B,Xconv_kernel)
        Yconv_kernel = tf.divide(-Y_filt , kernel_denom)
        Yconv_kernel = tf.where(self.cond,B,Yconv_kernel)
        self.Xconv_kernel = tf.reshape(Xconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1,1])
        self.Yconv_kernel = tf.reshape(Yconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1,1])
        x = tf.linspace(-1., 1., self.numpix_side)*self.im_side/2.
        y = tf.linspace(-1., 1., self.numpix_side)*self.im_side/2.
        self.Xim, self.Yim = tf.meshgrid(x, y)
        self.Xim = tf.reshape(self.Xim, [-1, self.numpix_side, self.numpix_side, 1])
        self.Yim = tf.reshape(self.Yim, [-1, self.numpix_side, self.numpix_side, 1])

    def get_deflection_angles(self, Kappa):
        #Calculate the Xsrc, Ysrc from the Xim, Yim for a given kappa map
        if (self.method=="conv2d"):
            alpha_x = tf.nn.conv2d(Kappa, self.Xconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi);
            alpha_y = tf.nn.conv2d(Kappa, self.Yconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi);
            Xsrc = tf.add(tf.reshape(self.Xim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_x )
            Ysrc = tf.add(tf.reshape(self.Yim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_y )
            
        if (self.method=="Unet"):
            alpha = self.RT (tf.identity(Kappa))
            alpha_x , alpha_y = tf.split(alpha,2,axis=3)
            #alpha = tf.identity(Kappa)
            #alpha_x  = alpha * 0.
            #alpha_y  = alpha * 0.
            Xsrc = tf.add(tf.reshape(self.Xim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_x )
            Ysrc = tf.add(tf.reshape(self.Yim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_y )

        return Xsrc, Ysrc , alpha_x , alpha_y


    def physical_model(self, Src , Kappa):

        Xsrc, Ysrc , _ , _ = self.get_deflection_angles(Kappa)
        
        IM = self.lens_source(Xsrc, Ysrc, Src)
        
#        Xsrc = tf.reshape(Xsrc, [-1, self.numpix_side, self.numpix_side, 1])
#        Ysrc = tf.reshape(Ysrc, [-1, self.numpix_side, self.numpix_side, 1])
#        Xsrc_pix, Ysrc_pix = self.coord_to_pix(Xsrc,Ysrc,0.,0., self.src_side ,self.numpix_side)
#        wrap = tf.reshape( tf.stack([Xsrc_pix, Ysrc_pix], axis = 3), [-1, self.numpix_side, self.numpix_side, 2])
#        IM = tf.contrib.resampler.resampler(Src, wrap)

        return IM
    
    def lens_source(self, Xsrc, Ysrc, Src):
        
        Xsrc = tf.reshape(Xsrc, [-1, self.numpix_side, self.numpix_side, 1])
        Ysrc = tf.reshape(Ysrc, [-1, self.numpix_side, self.numpix_side, 1])
        Xsrc_pix, Ysrc_pix = self.coord_to_pix(Xsrc,Ysrc,0.,0., self.src_side ,self.numpix_side)
        wrap = tf.reshape( tf.stack([Xsrc_pix, Ysrc_pix], axis = 3), [-1, self.numpix_side, self.numpix_side, 2])
        IM = tf.contrib.resampler.resampler(Src, wrap)

        return IM
        

    def simulate_noisy_lensed_image(self, Src , Kappa , noise_rms):
        IM = self.physical_model(Src , Kappa)
        #noise_rms = max_noise_rms#tf.random_uniform(shape=[1],minval=max_noise_rms/100.,maxval=max_noise_rms)
        noise = tf.random_normal(tf.shape(IM),mean=0.0,stddev = noise_rms)
        IM = IM + noise
        self.noise_rms = noise_rms
        return IM

    def coord_to_pix(self,X,Y,Xc,Yc,l,N):
        xmin = Xc-0.5*l
        ymin = Yc-0.5*l
        dx = l/(N-1.)
        i = tf.scalar_mul(1./dx, tf.math.add(X, -1.* xmin))
        j = tf.scalar_mul(1./dx, tf.math.add(Y, -1.* ymin))
        return i, j


def log_likelihood(Data,Model,noise_rms):
    #logL = 0.5 * tf.reduce_mean(tf.reduce_mean((Data - Model)**2, axis=2 ), axis=1 )
    logL = 0.5 * tf.reduce_mean(tf.reduce_mean((Data - Model)**2, axis=2 ), axis=1 ) / noise_rms**2
    return logL




def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))


class AUTOENCODER(tf.keras.Model):
    def __init__(self):
        super(AUTOENCODER, self).__init__()

        activation = lrelu4p
        D01 = tf.keras.layers.Conv2D(32 , (6, 6), activation=activation, strides=2 , padding='same') 
        D02 = tf.keras.layers.Conv2D(32 , (6, 6), activation=activation, strides=2 , padding='same') 
        D03 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, strides=2 , padding='same') 
        D04 = tf.keras.layers.Conv2D(1, (3, 3), activation=activation, strides=1 , padding='same') 
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
    
    def forward_pass(self,X):
        return self.decode(self.encode(X))
    
    def cost_function(self , X):
        prediction = self.forward_pass(X)
        cost = tf.reduce_mean((prediction - X)**2)
        return cost
        
    def train(self , X , optimizer):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            cost_value = self.cost_function(X)
        weight_grads = tape.gradient(cost_value, self.variables  )

        optimizer.apply_gradients(zip(weight_grads, self.variables), global_step=tf.train.get_or_create_global_step())
        return
    
    def Save(self):
        self.save_weights(self.AE_checkpoint_path)

    def Load(self):
        self.load_weights(self.AE_checkpoint_path)
        

def plot(samples):
    fig = plt.figure(figsize=(2*3, 3*3))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=None, hspace=None)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
        # plt.title(str(np.max(sample)))
    return fig
