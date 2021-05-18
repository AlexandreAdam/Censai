import tensorflow as tf
import tensorflow_addons as tfa
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, c
from scipy.constants import pi
import astropy.units as u
import math
import numpy as np

c = tf.constant(c, dtype=tf.float32)
pi = tf.constant(pi, dtype=tf.float32)
deg2mas = pi / 180 / 3600
sqrt2 = tf.constant(tf.sqrt(2.), dtype=tf.float32)


class Generator(tf.keras.utils.Sequence):
    """
    Class to generate source and kappa field during training and testing.
    #TODO check that this class wont work if kappa and source have different # of pixels
    """
    def __init__(
            self,
            total_items=1,
            batch_size=1,
            kappa_side_length=7.68,
            src_side_length=3.,
            kappa_side_pixels=48,
            src_side_pixels=48,
            z_source=2.,
            z_lens=0.5,
            train=True,
            norm=True):
        self.batch_size = batch_size
        self.total_items = total_items
        self.train = train
        self.norm = norm
        self.src_side_pixels = src_side_pixels

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, src_side_pixels)
        x = tf.cast(x, tf.float32)
        self.X_s, self.Y_s = [xx * src_side_length/2 for xx in tf.meshgrid(x, x)]
        self.X_k, self.Y_k = [xx * kappa_side_length/2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_side_length/2
        self.angular_diameter_distances(z_source, z_lens)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size) 

    def __getitem__(self, idx):
        return self.generate_batch()

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value * 1e6, dtype=tf.float32) # value in parsec
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value * 1e6, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value * 1e6, dtype=tf.float32)

    def generate_batch(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.01, maxval=0.6)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=2*pi)
        Rein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=2, maxval=4)

        #parameters for source
        sigma_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.5, maxval=1)
        x_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.5, maxval=0.5)
        y_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.5, maxval=0.5)

        kappa = self.kappa_field(xlens, ylens, elp, phi, Rein)
        source = self.gaussian_source(x_src, y_src, sigma_src)
        return source, kappa
 
    def kappa_field(self, xlens, ylens, elp, phi, Rein):
        """
        variables must have shape [batch_size, 1, 1] to be broadcastable
        """
        xk, yk = self.rotate(self.X_k, self.Y_k, phi)
        xs, ys = self.rotate(xlens, ylens, phi)
        A = self.dy_k/2*deg2mas
        r = tf.sqrt((xk - xs)**2 + ((yk - ys) * (1 - elp))**2 * deg2mas)
        # sigma_v = tf.sqrt(c**2/(4 * pi) * Rein * deg2mas * self.Ds / self.Dls) # velocity dispersion
        # Rein = 4 * pi * sigma_v**2 / c**2 * self.Dls / self.Ds  # Einstein radius
        kappa = tf.divide(tf.sqrt(1 - elp) * Rein, 2 * r)

        # normalize mass at the center
        mass_inside_center_pixel = 2 * A * (tf.math.log(sqrt2 + 1) - tf.math.log(sqrt2 * A - A) + tf.math.log(3 * A + 2*sqrt2 * A))
        density_center = tf.sqrt(1 - elp) * Rein/2 * mass_inside_center_pixel / (2 * A)**2
        density_center = tf.reshape(density_center, [-1]) # remove singleton dimensions for sparse update

        ind = tf.argmin(tf.reshape(r, [self.batch_size, -1]), axis=1)
        # broadcast index array to batch dim
        ind = [[i, ind[i]] for i in range(self.batch_size)]
        kappa = tf.reshape(kappa, [self.batch_size, -1])
        # sparse update of kappa
        density_center = tf.tensor_scatter_nd_add(kappa, ind, density_center)
        kappa = tf.reshape(kappa, r.shape)

        return kappa 

    @staticmethod
    @tf.function
    def rotate(x, y, phi):
        """
        Rotate the coordinate system by phi. 
        """
        rho = tf.sqrt(x**2 + y**2)
        theta = tf.math.atan2(y, x) - phi
        x_prime = rho * tf.math.cos(theta)
        y_prime = rho * tf.math.sin(theta)
        return x_prime, y_prime # shape [batch_size, pixel, pixel]

    def gaussian_source(self, x, y, sigma):
        rho_squared = (self.X_s - x)**2 + (self.Y_s - y)**2
        im = np.exp(-0.5 * rho_squared / sigma**2)
        if self.norm:
            im /= tf.reduce_max(im)
        return im


class NISGenerator(tf.keras.utils.Sequence):
    """
    Generate pairs of kappa field and deflection angles of pseudo-elliptical lensing potential 
    of a Non-Singular Isothermal Sphere density profile lens.
    """
    def __init__(self, 
            total_items=1, 
            batch_size=1, 
            kappa_side_length=7.68, 
            src_side_length=3., 
            pixels=128, 
            x_c=0.001, # arcsec
            z_source=2., 
            z_lens=0.5, 
            train=True, 
            norm=True,
            model="raytracer", # alternative is rim
            method="analytic"
            ):
        """
        :param kappa_side_length: Angular FOV of lens plane field (in mas)
        :param src_side_length: Angular FOV of source plane (in mas) -> should be much smaller than kappa FOV
        """
        self.batch_size = batch_size
        self.total_items = total_items
        self.train = train
        self.norm = norm
        self.model = model.lower()
        self._kap_side = kappa_side_length
        self._src_side = src_side_length
        self.pixels = pixels
        self.method = method

        self.x_c = tf.constant(x_c, tf.float32)

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, pixels)
        x = tf.cast(x, tf.float32)
        self.x_source, self.y_source = [xx * src_side_length/2 for xx in tf.meshgrid(x, x)]
        self.theta1, self.theta2 = [xx * kappa_side_length/2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_side_length/2
        self.angular_diameter_distances(z_source, z_lens)
        self.set_deflection_angles_vars()

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self, method):
        if method in ["analytic", "conv2d", "approximate"]:
            self.__method = method
        else:
            raise NotImplementedError(method)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size) 

    def __getitem__(self, idx):
        if self.model == "raytracer":
            return self.generate_batch()
        elif self.model == "rim":
            return self.generate_batch_rim()

    def generate_batch(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=0.2)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        Rein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1, maxval=2.5)

        kappa = self.kappa_field(xlens, ylens, elp, phi, Rein)
        if self.method == "analytic":
            alpha = self.analytical_deflection_angles(xlens, ylens, elp, phi, Rein)
        elif self.method == "approximate":
            alpha = self.approximate_deflection_angles(xlens, ylens, elp, phi, Rein)
        elif self.method == "conv2d":
            alpha = self.get_deflection_angles(kappa)
        return kappa, alpha #(X, Y)

    def generate_batch_rim(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)

        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-.5, maxval=.5)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-.5, maxval=.5)
        elp   = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=0.2)
        phi   = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        r_ein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1, maxval=2.)

        xs    = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.1, maxval=0.1)
        ys    = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.1, maxval=0.1)
        e     = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=0.3)
        phi_s = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        w     = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.1, maxval=0.2)

        kappa = self.kappa_field(xlens, ylens, elp, phi, r_ein)
        source = self.source_model(xs, ys, e, phi_s, w)
        lensed_image = self.lens_source(source, r_ein, elp, phi, xlens, ylens)
        return lensed_image, source, kappa #(X, Y1, Y2)

    def lens_source(self, source, r_ein, e, phi, x0, y0):
        if self.method == "analytic":
            alpha = self.analytical_deflection_angles(x0, y0, e, phi, r_ein)
        elif self.method == "approximate":
            alpha = self.approximate_deflection_angles(x0, y0, e, phi, r_ein)
        elif self.method == "conv2d":
            kap = self.kappa_field(x0, y0, e, phi, r_ein)
            alpha = self.conv2d_deflection_angles(kap)
        # place the origin at the center of mass
        theta1, theta2 = self.theta1 - x0, self.theta2 - x0
        # lens equation
        beta1 = theta1 - alpha[..., 0]
        beta2 = theta2 - alpha[..., 1]
        # back to pixel grid
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        wrap = tf.stack([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, wrap)  # bilinear interpolation
        return im

    def src_coord_to_pix(self, x, y):
        dx = self._src_side/(self.pixels - 1)
        xmin = -0.5 * self._src_side
        ymin = -0.5 * self._src_side
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def source_model(self, x0, y0, elp, phi, w): # for rim, simple gaussian for testing the model
        beta1 = self.x_source - x0
        beta2 = self.y_source - y0
        _beta1 = beta1 * np.cos(phi) + beta2 * np.sin(phi)
        _beta2 = -beta1 * np.sin(phi) + beta2 * np.cos(phi) 
        rho_sq = _beta1**2/(1-elp) + _beta2**2 * (1 - elp)
        source = np.exp(-0.5 * rho_sq / w**2) #/ 2 / np.pi / w**2
        return source[..., tf.newaxis] # add channel dimension

    def kappa_field(self, xlens, ylens, elp, phi, r_ein):
        """
        :param xlens: Horizontal position of the lens  (in mas)
        :param ylens: Vertical position of the lens (in mas)
        :param x_c: Critical radius (in mas) where the density is flattened to avoid the singularity
        """
        xk, yk = self.rotated_and_shifted_coords(xlens, ylens, phi)
        kappa = 0.5 * r_ein / (xk**2/(1-elp) + yk**2*(1-elp) + self.x_c**2)**(1/2)
        return kappa[..., tf.newaxis]  # add channel dimension

    def approximate_deflection_angles(self, xlens, ylens, elp, phi, r_ein):
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(xlens, ylens, phi)
        denominator = (theta1**2/(1-elp) + theta2**2*(1-elp) + self.x_c**2)**(1/2)
        alpha1 = r_ein * theta1 / (1 - elp) / denominator
        alpha2 = r_ein * theta2 * (1 - elp) / denominator
        # rotate back to original orientation of coordinate system
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return tf.stack([alpha1, alpha2], axis=-1)  # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    def analytical_deflection_angles(self, xlens, ylens, elp, phi, r_ein):
        b, q, s = self._param_conv(elp, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(xlens, ylens, -phi)
        psi = tf.sqrt(q ** 2 * (s**2 + theta1**2) + theta2**2)
        alpha1 = b / tf.sqrt(1. - q ** 2) * tf.math.atan(np.sqrt(1. - q ** 2) * theta1 / (psi + s))
        alpha2 = b / tf.sqrt(1. - q ** 2) * tf.math.atanh(np.sqrt(1. - q ** 2) * theta2 / (psi + s*q**2))
        # rotate back
        alpha1, alpha2 = self._rotate(alpha1, alpha2, phi)
        return tf.stack([alpha1, alpha2], axis=-1)

    def conv2d_deflection_angles(self, Kappa):
        alpha_x = tf.nn.conv2d(Kappa, self.Xconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi);
        alpha_y = tf.nn.conv2d(Kappa, self.Yconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi);
        return tf.concat([alpha_x, alpha_y], axis=-1)

    def rotated_and_shifted_coords(self, x0, y0, phi):
        ###
        # Important to shift then rotate, we move to the point of view of the
        # lens before rotating the lens (rotation and translation are not commutative).
        ###
        theta1 = self.theta1 - x0
        theta2 = self.theta2 - y0
        return self._rotate(theta1, theta2, phi)

    def _rotate(self, x, y, angle):
        return x * tf.math.cos(angle) + y * tf.math.sin(angle), -x * tf.math.sin(angle) + y * tf.math.cos(angle)

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value, dtype=tf.float32) # value in Mpc
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value, dtype=tf.float32)
                
    def set_deflection_angles_vars(self):
        self.kernel_side_l = 2 * self.pixels + 1  # this number should be odd
        self.cond = np.zeros((self.kernel_side_l, self.kernel_side_l))
        self.cond[self.pixels, self.pixels] = True
        self.dx_kap = self._kap_side / (self.pixels - 1)
        x = tf.linspace(-1., 1., self.kernel_side_l) * self._kap_side
        y = tf.linspace(-1., 1., self.kernel_side_l) * self._kap_side
        X_filt, Y_filt = tf.meshgrid(x, y)
        kernel_denom = tf.square(X_filt) + tf.square(Y_filt)
        Xconv_kernel = tf.divide(-X_filt, kernel_denom)
        B = tf.zeros_like(Xconv_kernel)
        Xconv_kernel = tf.where(self.cond, B, Xconv_kernel)
        Yconv_kernel = tf.divide(-Y_filt, kernel_denom)
        Yconv_kernel = tf.where(self.cond, B, Yconv_kernel)
        self.Xconv_kernel = tf.reshape(Xconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1, 1])
        self.Yconv_kernel = tf.reshape(Yconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1, 1])
        x = tf.linspace(-1., 1., self.pixels) * self._src_side/2. #TODO make options to have different size source image
        y = tf.linspace(-1., 1., self.pixels) * self._src_side/2.
        self.Xim, self.Yim = tf.meshgrid(x, y)
        self.Xim = tf.reshape(self.Xim, [-1, self.pixels, self.pixels, 1])
        self.Yim = tf.reshape(self.Yim, [-1, self.pixels, self.pixels, 1])

    def _param_conv(self, elp, r_ein):
        q = (1 - elp) / (1 + elp) # axis ratio
        r_ein_conv = 2. * q * r_ein / tf.sqrt(1.+q**2)
        b = r_ein_conv * tf.sqrt((1 + q ** 2) / 2)
        s = self.x_c * tf.sqrt((1 + q**2) / (2*q**2))
        return b, q, s


class SRC_KAPPA_Generator(object):
    def __init__(self, train_batch_size=1, test_batch_size=1, kap_side_length=7.68, src_side=3.0, num_src_side=48,
                 num_kappa_side=48):
        self.src_side = src_side
        self.kap_side_length = kap_side_length
        self.num_src_side = num_src_side
        self.num_kappa_side = num_kappa_side
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        x_s = np.linspace(-1., 1., self.num_src_side, dtype='float32') * self.src_side / 2
        y_s = np.linspace(-1., 1., self.num_src_side, dtype='float32') * self.src_side / 2
        self.dy_s = y_s[1] - y_s[0]
        self.Xsrc, self.Ysrc = np.meshgrid(x_s, y_s)
        x_k = np.linspace(-1., 1., self.num_kappa_side, dtype='float32') * self.kap_side_length / 2
        y_k = np.linspace(-1., 1., self.num_kappa_side, dtype='float32') * self.kap_side_length / 2
        self.dy_k = y_k[1] - y_k[0]
        self.Xkap, self.Ykap = np.meshgrid(x_k, y_k)
        self.Kappa_tr = np.zeros((train_batch_size, num_kappa_side, num_kappa_side, 1), dtype='float32')
        self.Source_tr = np.zeros((train_batch_size, num_src_side, num_src_side, 1), dtype='float32')
        self.Kappa_ts = np.zeros((test_batch_size, num_kappa_side, num_kappa_side, 1), dtype='float32')
        self.Source_ts = np.zeros((test_batch_size, num_src_side, num_src_side, 1), dtype='float32')

    def Kappa_fun(self, xlens, ylens, elp, phi, Rein, rc=0, Zlens=0.5, Zsource=2.0, c=299800000):
        Dds = cosmo.angular_diameter_distance_z1z2(Zlens, Zsource).value * 1e6
        Ds = cosmo.angular_diameter_distance(Zsource).value * 1e6
        sigma_v = np.sqrt(c ** 2 / (4 * np.pi) * Rein * np.pi / 180 / 3600 * Ds / Dds)
        A = self.dy_k / 2. * (2 * np.pi / (360 * 3600))
        rcord, thetacord = np.sqrt(self.Xkap ** 2 + self.Ykap ** 2), np.arctan2(self.Ykap, self.Xkap)
        thetacord = thetacord - phi
        Xkap, Ykap = rcord * np.cos(thetacord), rcord * np.sin(thetacord)
        rlens, thetalens = np.sqrt(xlens ** 2 + ylens ** 2), np.arctan2(ylens, xlens)
        thetalens = thetalens - phi
        xlens, ylens = rlens * np.cos(thetalens), rlens * np.sin(thetalens)
        r = np.sqrt((Xkap - xlens) ** 2 + ((Ykap - ylens) * (1 - elp)) ** 2) * (2 * np.pi / (360 * 3600))
        Rein = (4 * np.pi * sigma_v ** 2 / c ** 2) * Dds / Ds
        kappa = np.divide(np.sqrt(1 - elp) * Rein, (2 * np.sqrt(r ** 2 + rc ** 2)))
        mass_inside_00_pix = 2. * A * (np.log(2. ** (1. / 2.) + 1.) - np.log(2. ** (1. / 2.) * A - A) + np.log(
            3. * A + 2. * 2. ** (1. / 2.) * A))
        density_00_pix = np.sqrt(1. - elp) * Rein / (2.) * mass_inside_00_pix / ((2. * A) ** 2.)
        ind = np.argmin(r)
        kappa.flat[ind] = density_00_pix
        return kappa

    def gen_source(self, x_src=0, y_src=0, sigma_src=1, norm=False):
        Im = np.exp(-(((self.Xsrc - x_src) ** 2 + (self.Ysrc - y_src) ** 2) / (2. * sigma_src ** 2)))
        if norm is True:
            Im = Im / np.max(Im)
        return Im

    def draw_k_s(self, train_or_test):
        if (train_or_test == "train"):
            np.random.seed(seed=None)
            num_samples = self.train_batch_size
        if (train_or_test == "test"):
            np.random.seed(seed=136)
            num_samples = self.test_batch_size

        for i in range(num_samples):
            # parameters for kappa
            # np.random.seed(seed=155)
            xlens = np.random.uniform(low=-1.0, high=1.)
            ylens = np.random.uniform(low=-1.0, high=1.)
            elp = np.random.uniform(low=0.01, high=0.6)
            phi = np.random.uniform(low=0.0, high=2. * np.pi)
            Rein = np.random.uniform(low=2.0, high=4.0)

            # parameters for source
            sigma_src = np.random.uniform(low=0.5, high=1.0)
            x_src = np.random.uniform(low=-0.5, high=0.5)
            y_src = np.random.uniform(low=-0.5, high=0.5)
            norm_source = True

            if (train_or_test == "train"):
                self.Source_tr[i, :, :, 0] = self.gen_source(x_src=x_src, y_src=y_src, sigma_src=sigma_src,
                                                             norm=norm_source)
                self.Kappa_tr[i, :, :, 0] = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
            if (train_or_test == "test"):
                self.Source_ts[i, :, :, 0] = self.gen_source(x_src=x_src, y_src=y_src, sigma_src=sigma_src,
                                                             norm=norm_source)
                self.Kappa_ts[i, :, :, 0] = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
        return

    def draw_average_k_s(self):
        src = self.gen_source(x_src=0., y_src=0., sigma_src=0.5, norm=True)
        kappa = self.Kappa_fun(0., 0., 0.02, 0., 3.0)
        src = src.reshape(1, self.num_src_side, self.num_src_side, 1)
        kappa = kappa.reshape(1, self.num_kappa_side, self.num_kappa_side, 1)

        src = np.repeat(src, self.train_batch_size, axis=0)
        kappa = np.repeat(kappa, self.train_batch_size, axis=0)
        return src, kappa