import tensorflow as tf
import numpy as np
from scipy import interpolate




    
class Likelihood(object):
    '''
    This class will contain the
    likelihood that will be fed to the RIM

    '''
    #img_pl,lens_pl,noise,noise_cov
    def __init__(self, im_side= 7.68, src_side=3.0, numpix_side = 192):
        '''
        Initialize the object.
        '''
        
        self.im_side = im_side 
        self.numpix_side = numpix_side
        self.src_side     = src_side


    def get_deflection_angles(self, Xim, Yim, Kappa, kap_cent, kap_side):
        #Calculate the Xsrc, Ysrc from the Xim, Yim for a given kappa map
        
        kap_numpix = (Kappa.shape.as_list())[1]
        kernel_side_l = kap_numpix*2+1;
        
        cond = np.zeros((kernel_side_l,kernel_side_l))
        cond[kap_numpix,kap_numpix] = True

        dx_kap = kap_side/(kap_numpix-1)
        
        x = tf.linspace(-1., 1., kernel_side_l)*kap_side
        y = tf.linspace(-1., 1., kernel_side_l)*kap_side
        X_filt, Y_filt = tf.meshgrid(x, y)
        
#         tf.add_to_collection('xfilt', X_filt)
#         tf.add_to_collection('yfilt', Y_filt)
        
        kernel_denom = tf.square(X_filt) + tf.square(Y_filt)
        Xconv_kernel = tf.divide(-X_filt , kernel_denom) 
        
        B = tf.zeros_like(Xconv_kernel)
        Xconv_kernel = tf.where(cond,B,Xconv_kernel)

        Yconv_kernel = tf.divide(-Y_filt , kernel_denom) 

        Yconv_kernel = tf.where(cond,B,Yconv_kernel)

#         tf.add_to_collection('xker', Xconv_kernel)
#         tf.add_to_collection('yker', Yconv_kernel)
        
        Xconv_kernel = tf.reshape(Xconv_kernel, [kernel_side_l, kernel_side_l, 1,1])
        Yconv_kernel = tf.reshape(Yconv_kernel, [kernel_side_l, kernel_side_l, 1,1])
        
#         tf.add_to_collection('xker2', Xconv_kernel)
#         tf.add_to_collection('yker2', Yconv_kernel)
        
        alpha_x = tf.nn.conv2d(Kappa, Xconv_kernel, [1, 1, 1, 1], "SAME") * (dx_kap**2/np.pi);
        alpha_y = tf.nn.conv2d(Kappa, Yconv_kernel, [1, 1, 1, 1], "SAME") * (dx_kap**2/np.pi);
        
#        tf.add_to_collection('alpha_x', alpha_x)
#        tf.add_to_collection('alpha_y', alpha_y)
        
        #X_kap = tf.linspace(-0.5, 0.5, kap_numpix)*kap_side/1.
        #Y_kap = tf.linspace(-0.5, 0.5, kap_numpix)*kap_side/1.
        #Xkap, Ykap = tf.meshgrid(X_kap, Y_kap)
        
        Xim = tf.reshape(Xim, [-1, self.numpix_side, self.numpix_side, 1])
        Yim = tf.reshape(Yim, [-1, self.numpix_side, self.numpix_side, 1])
        
        
        x_centshif = -(kap_cent[0]*(1./dx_kap))*tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32) 
        x_centshif = tf.reshape(x_centshif, [-1, self.numpix_side, self.numpix_side, 1])
        x_resize = tf.scalar_mul( (1./dx_kap), tf.math.add(Xim, 0.5*kap_side*tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32)) )
        x_resize = tf.reshape(x_resize, [-1, self.numpix_side, self.numpix_side, 1])
        
        Xim_pix = tf.math.add( x_centshif , x_resize )  
        
        
        
        y_centshif = -(kap_cent[1]*(1./dx_kap))*tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32) 
        y_centshif = tf.reshape(y_centshif, [-1, self.numpix_side, self.numpix_side, 1])
        y_resize = tf.scalar_mul( (1./dx_kap), tf.math.add(Yim, 0.5*kap_side*tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32)) )
        y_resize = tf.reshape(y_resize, [-1, self.numpix_side, self.numpix_side, 1])
        
        Yim_pix = tf.math.add( y_centshif , y_resize )  
        
        
        
        Xim_pix = tf.reshape(Xim_pix ,  [-1, self.numpix_side, self.numpix_side, 1])
        Yim_pix = tf.reshape(Yim_pix ,  [-1, self.numpix_side, self.numpix_side, 1])
        
        
        
        wrap = tf.reshape( tf.stack([Xim_pix, Yim_pix], axis = 3), [-1, self.numpix_side, self.numpix_side, 2])
        
        
        
        A = tf.constant(1)
        mult = tf.stack([tf.shape(alpha_x)[0], A,A,A])
        
        batch_wrap = tf.manip.tile(wrap, mult)
        
        
        
        alphax_interp = tf.contrib.resampler.resampler(alpha_x, batch_wrap)
        alphay_interp = tf.contrib.resampler.resampler(alpha_y, batch_wrap)
        
        Xsrc = tf.math.add(tf.reshape(Xim, [-1, self.numpix_side, self.numpix_side, 1]),  -alphax_interp )
        Ysrc = tf.math.add(tf.reshape(Yim, [-1, self.numpix_side, self.numpix_side, 1]),  -alphay_interp )
        
#         tf.add_to_collection('Xsrc', Xsrc)
#         tf.add_to_collection('Ysrc', Ysrc)
        
        return Xsrc, Ysrc
    
    def get_lensed_image(self, Kappa, kap_cent, kap_side, Src, noisy=True, max_noise_rms=0.05):
        
        x = tf.linspace(-1., 1., self.numpix_side)*self.im_side/2.
        y = tf.linspace(-1., 1., self.numpix_side)*self.im_side/2.
        Xim, Yim = tf.meshgrid(x, y)
        
        
        Xsrc, Ysrc = self.get_deflection_angles(Xim, Yim, Kappa, kap_cent, kap_side)
        
        Xsrc = tf.reshape(Xsrc, [-1, self.numpix_side, self.numpix_side, 1])
        Ysrc = tf.reshape(Ysrc, [-1, self.numpix_side, self.numpix_side, 1])
                
        Xsrc_pix, Ysrc_pix = self.coord_to_pix(Xsrc,Ysrc,0.,0., self.src_side ,self.numpix_side)
        
#        tf.add_to_collection('Xsrc', Xsrc)
#        tf.add_to_collection('Ysrc', Ysrc)
#
#        
#        tf.add_to_collection('Xsrc_pix', Xsrc_pix)
#        tf.add_to_collection('Ysrc_pix', Ysrc_pix)
        
        wrap = tf.reshape( tf.stack([Xsrc_pix, Ysrc_pix], axis = 3), [-1, self.numpix_side, self.numpix_side, 2])
        
        
        IM = tf.contrib.resampler.resampler(Src, wrap)
        
        if noisy == True:
            noise_rms = max_noise_rms#tf.random_uniform(shape=[1],minval=max_noise_rms/100.,maxval=max_noise_rms)
            noise = tf.random_normal(tf.shape(IM),mean=0.0,stddev = noise_rms)
            IM = tf.add(IM,noise)
            self.noise_rms = noise_rms
        
        return IM
    


    def coord_to_pix(self,X,Y,Xc,Yc,l,N):
    
        xmin = Xc-0.5*l
        ymin = Yc-0.5*l
        dx = l/(N-1.)

        j = tf.scalar_mul(1./dx, tf.math.add(X, -1.* xmin))
        i = tf.scalar_mul(1./dx, tf.math.add(Y, -1.* ymin))
        
        return i, j
        
    def Loglikelihood(self, src, Kappa, kap_cent, kap_side, noisy=True, max_noise_rms=0.05):
        '''
        Computes the likelihood of observing an image (img), given a model image for the src
        (src).  Computes the raytracing over multiple images in a vectorized way.  
        '''

        img_pred = self.get_lensed_image(Kappa, kap_cent, kap_side, src,noisy=False, max_noise_rms=max_noise_rms)
        #tf.add_to_collection('mykappa', kap_cent)
        #xsrc , ysrc = self.raytrace2()

        #xsrc = tf.reshape(xsrc,[-1])
        #ysrc = tf.reshape(ysrc,[-1])

        #img_pred = self._interpolate(self.src,xsrc,ysrc,[self.numpix_side,self.numpix_side],self.src_res)
        img_pred = tf.reshape(img_pred,tf.shape(self.trueimage))

#        if self.psf:
#            img_pred = tf.nn.conv2d(img_pred,self.psf_pl,strides=[1,1,1,1],padding='SAME')
#
#        if hasattr(self,'mask_pl'):
#            img_pred = tf.multiply(img_pred,self.mask_pl)
        mse = tf.reduce_sum(tf.square(tf.subtract(img_pred,self.trueimage)))

        if hasattr(self,'noise_rms'):
            mse /= tf.square(self.noise_rms)

        return mse     
