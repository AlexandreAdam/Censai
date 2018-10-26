import tensorflow as tf
import numpy as np
from scipy import interpolate

def Kappa_fun(xlens, ylens, elp, phi, sigma_v, numkappa_side = 193, kap_side_length = 2, rc=0, Ds = 1753486987.8422, Dds = 1125770220.58881, c = 299800000):
    
    x = np.linspace(-1, 1, numkappa_side) * kap_side_length/2
    y = np.linspace(-1, 1, numkappa_side) * kap_side_length/2
    xv, yv = np.meshgrid(x, y)
    
    A = (y[1]-y[0])/2. *(2*np.pi/ (360*3600) )
    
    rcord, thetacord = np.sqrt(xv**2 + yv**2) , np.arctan2(xv, yv)
    thetacord = thetacord - phi
    Xkap, Ykap = rcord*np.cos(thetacord), rcord*np.sin(thetacord)
    
    rlens, thetalens = np.sqrt(xlens**2 + ylens**2) , np.arctan2(xlens, ylens)
    thetalens = thetalens - phi
    xlens, ylens = rlens*np.cos(thetalens), rlens*np.sin(thetalens)
    
    r = np.sqrt((Xkap-xlens)**2 + ((Ykap-ylens) * (1-elp) )**2) *(2*np.pi/ (360*3600) )
    
    Rein = (4*np.pi*sigma_v**2/c**2) * Dds /Ds 
    
    kappa = np.divide( np.sqrt(1-elp)* Rein ,  (2* np.sqrt( r**2 + rc**2)))
    
    mass_inside_00_pix = 2.*A*(np.log(2.**(1./2.) + 1.) - np.log(2.**(1./2.)*A - A) + np.log(3.*A + 2.*2.**(1./2.)*A))
    
    print A
    print mass_inside_00_pix
    
    density_00_pix = np.sqrt(1.-elp) * Rein/(2.) * mass_inside_00_pix/((2.*A)**2.)
    
    print density_00_pix
    
    ind = np.argmin(r)
    
    kappa.flat[ind] = density_00_pix
    
    return kappa

    
class Likelihood(object):
    '''
    This class will contain the
    likelihood that will be fed to the RIM

    '''
    #img_pl,lens_pl,noise,noise_cov
    def __init__(self, pix_to_rad=0.04, numpix_side = 192):
        '''
        Initialize the object.  Lets have img_pl be the shape we expect to be fed to the network [m,N,N,1]
        and do transposing to reshape things as we need.
        '''
        
        self.pix_to_rad = pix_to_rad
        self.numpix_side = numpix_side


    def get_deflection_angles(self, Xim, Yim, Kappa):
        #Calculate the Xsrc, Ysrc from the Xim, Yim for a given kappa map
        
        
        x = tf.linspace(-2., 2., self.numpix_side*2)*self.pix_to_rad
        y = tf.linspace(-2., 2., self.numpix_side*2)*self.pix_to_rad
        X_filt, Y_filt = tf.meshgrid(x, y)
        
        kernel_denom = tf.square(X_filt) + tf.square(Y_filt)
        Xconv_kernel = tf.divide(X_filt , kernel_denom) 
        Yconv_kernel = tf.divide(Y_filt , kernel_denom) 
        
        Xconv_kernel = tf.reshape(Xconv_kernel, [self.numpix_side*2, self.numpix_side*2, 1,1])
        Yconv_kernel = tf.reshape(Yconv_kernel, [self.numpix_side*2, self.numpix_side*2, 1,1])
        
        Xsrc = tf.math.add(tf.reshape(Xim, [1, self.numpix_side, self.numpix_side, 1]), tf.nn.conv2d(Kappa, Xconv_kernel, [1, 1, 1, 1], "SAME"))
        Ysrc = tf.math.add(tf.reshape(Yim, [1, self.numpix_side, self.numpix_side, 1]), tf.nn.conv2d(Kappa, Yconv_kernel, [1, 1, 1, 1], "SAME"))
        
        return Xsrc, Ysrc
    
    def get_lensed_image(self, Kappa, Src):
        
        x = tf.linspace(-1., 1., self.numpix_side)*self.pix_to_rad
        y = tf.linspace(-1., 1., self.numpix_side)*self.pix_to_rad
        Xim, Yim = tf.meshgrid(x, y)
        
        if Kappa is None:
            self.build_kappa()
        Xsrc, Ysrc = self.get_deflection_angles(Xim, Yim, Kappa)
        
        Xsrc = tf.reshape(Xsrc, [-1, self.numpix_side, self.numpix_side, 1])
        Ysrc = tf.reshape(Ysrc, [-1, self.numpix_side, self.numpix_side, 1])
        
        Xsrc_pix = tf.scalar_mul( self.numpix_side/2, tf.math.add(Xsrc, tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32)) )
        Ysrc_pix = tf.scalar_mul( self.numpix_side/2, tf.math.add(Ysrc, tf.ones([1, self.numpix_side, self.numpix_side, 1], dtype=tf.float32)) )
        
        wrap = tf.reshape( tf.stack([Xsrc_pix, Ysrc_pix], axis = 3), [1, self.numpix_side, self.numpix_side, 2])
        
        
        IM = tf.contrib.resampler.resampler(Src, wrap)
        
        return IM
    
