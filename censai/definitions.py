import tensorflow as tf
import numpy as np

DTYPE = tf.float32


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus( -x -5.0 )


def xsquared(x):
    return (x/4)**2


def logKap_normalization(logkappa , dir="code"):
    if dir=="code":
        return tf.nn.relu(logkappa + 4.0) / 7.0
    else:
        return (logkappa * 7.0 - 4.0)


def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator


def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))




class LensUtil:
    def __init__(self, im_side=7.68, src_side=3.0, numpix_side=256 , kap_side=7.68 , method="conv2d"):
        self.im_side = im_side
        self.numpix_side = numpix_side
        self.src_side = src_side
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
            alpha = self.RT(tf.identity(Kappa))
            alpha_x , alpha_y = tf.split(alpha,2,axis=3)
            #alpha = tf.identity(Kappa)
            #alpha_x  = alpha * 0.
            #alpha_y  = alpha * 0.
            Xsrc = tf.add(tf.reshape(self.Xim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_x )
            Ysrc = tf.add(tf.reshape(self.Yim, [-1, self.numpix_side, self.numpix_side, 1]),  - alpha_y )

        return Xsrc, Ysrc , alpha_x , alpha_y


    def physical_model(self, Src , logKappa):

        #logKappa = logKap_normalization( logKappa , dir="decode")
        Kappa = 10.0 ** logKappa
        Xsrc, Ysrc , _ , _ = self.get_deflection_angles(Kappa)
        
        IM = self.lens_source(Xsrc, Ysrc, Src)
        
        Xsrc = tf.reshape(Xsrc, [-1, self.numpix_side, self.numpix_side, 1])
        Ysrc = tf.reshape(Ysrc, [-1, self.numpix_side, self.numpix_side, 1])
        Xsrc_pix, Ysrc_pix = self.coord_to_pix(Xsrc,Ysrc,0.,0., self.src_side ,self.numpix_side)
        wrap = tf.reshape( tf.stack([Xsrc_pix, Ysrc_pix], axis = 3), [-1, self.numpix_side, self.numpix_side, 2])
        IM = tf.contrib.resampler.resampler(Src, wrap)

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


# def plot(samples):
#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(2*3, 3*3))
#     gs = gridspec.GridSpec(3, 2)
#     gs.update(wspace=None, hspace=None)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample)
#         # plt.title(str(np.max(sample)))
#     return fig
