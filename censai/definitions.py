import tensorflow as tf
import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

DTYPE = tf.float32


def theta_einstein(kappa, rescaling, physical_pixel_scale, sigma_crit, Dds, Ds, Dd):
    """ #TODO deal with astropy units internally in this function
    Einstein radius is computed with the mass inside the Einstein ring, which corresponds to
    where kappa > 1.
    Args:
        kappa: A single kappa map of shape [crop_pixels, crop_pixels, channels]
        rescaling: Possibly an array of rescaling factor of the kappa maps
        physical_pixel_scale: Pixel scale in Comoving Mpc for the kappa map grid (in Mpc/pixels, with astropy units)
        sigma_crit: Critical Density (should be in units kg / Mpc**2 with astropy units attached)
        Dds: Angular diameter distance between the deflector (d) and the source (s) (in Mpc with astropy units attached)
        Ds: Augular diameter distance between the observer and the source (in Mpc with astropy units attached)
        Dd: Augular diameter distance between the observer and the deflector (in Mpc with astropy units attached)

    Returns: Einstein radius in arcsecond (an array of floats, no astropy units attached)
    """
    rescaling = np.atleast_1d(rescaling)
    kap = rescaling[..., np.newaxis, np.newaxis, np.newaxis] * kappa[np.newaxis, ...]
    mass_inside_einstein_radius = np.sum(kap * (kap > 1), axis=(1, 2, 3)) * sigma_crit * physical_pixel_scale**2
    return (np.sqrt(4 * G / c ** 2 * mass_inside_einstein_radius * Dds / Ds / Dd).decompose() * u.rad).to(u.arcsec).value


def compute_rescaling_probabilities(kappa, rescaling_array, physical_pixel_scale, sigma_crit, Dds, Ds, Dd,
                                    bins=10, min_theta_e=1., max_theta_e=5.,
                                    ):
    """
    Args:
        kappa: A single kappa map, of shape [crop_pixels, crop_pixels, channel]
        rescaling_array: An array of rescaling factor
        bins: Number of bins of the histogram used to figure out einstein radius distribution
        min_theta_e: Minimum desired value of Einstein ring radius (in arcsec)
        max_theta_e: Maximum desired value of Einstein ring radius (in arcsec)

    Returns: Probability of picking rescaling factor in rescaling array so that einstein radius has a
        uniform distribution between minimum and maximum allowed value
    """
    p = np.zeros_like(rescaling_array)
    theta_e = theta_einstein(kappa, rescaling_array, physical_pixel_scale, sigma_crit, Dds, Ds, Dd)
    # compute theta distribution
    select = (theta_e >= min_theta_e) & (theta_e <= max_theta_e)
    theta_hist, bin_edges = np.histogram(theta_e, bins=bins, range=[min_theta_e, max_theta_e], density=False)
    # for each theta_e, find bin index of our histogram. We give the left edges of the bin (param right=False)
    rescaling_bin = np.digitize(theta_e[select], bin_edges[:-1],
                                right=False) - 1  # bin 0 is outside the range to the left by default
    theta_hist[theta_hist == 0] = 1  # give empty bins a weight
    p[select] = 1 / theta_hist[rescaling_bin]
    p /= p.sum()  # normalize our new probability distribution
    return p


def belu(x):
    """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
    x_shape = x.shape
    x1, x2 = tf.split(tf.reshape(x, x_shape[:-1] + [-1, 2]), 2, axis=-1)
    y1 = tf.nn.elu(x1)
    y2 = -tf.nn.elu(-x2)
    return tf.reshape(tf.concat([y1, y2], axis=-1), x_shape)


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


def to_float(x):
    """Cast x to float; created because tf.to_float is deprecated."""
    return tf.cast(x, tf.float32)


def inverse_exp_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
    inv_base = tf.exp(tf.math.log(min_value) / float(max_step))
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    return inv_base**tf.maximum(float(max_step) - step, 0.0)


def inverse_lin_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    progress = tf.minimum(step / float(max_step), 1.0)
    return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)

    def sigmoid(x):
        return 1 / (1 + tf.exp(-x))

    def inv_sigmoid(y):
        return tf.math.log(y / (1 - y))

    assert min_value > 0, (
          "sigmoid's output is always >0 and <1. min_value must respect "
          "these bounds for interpolation to work.")
    assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

    # Find
    #   x  s.t. sigmoid(x ) = y_min and
    #   x' s.t. sigmoid(x') = y_max
    # We will map [0, max_step] to [x_min, x_max].
    y_min = min_value
    y_max = 1.0 - min_value
    x_min = inv_sigmoid(y_min)
    x_max = inv_sigmoid(y_max)

    x = tf.minimum(step / float(max_step), 1.0)  # [0, 1]
    x = x_min + (x_max - x_min) * x  # [x_min, x_max]
    y = sigmoid(x)  # [y_min, y_max]

    y = (y - y_min) / (y_max - y_min)  # [0, 1]
    y = y * (1.0 - y_min)  # [0, 1-y_min]
    y += y_min  # [y_min, 1]
    return y



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
