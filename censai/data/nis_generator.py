import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
from censai.definitions import COSMO
from astropy.constants import G, c, M_sun
from astropy import units as u

pi = np.pi


class NISGenerator(tf.keras.utils.Sequence):
    """
    Generate pairs of kappa field and deflection angles of pseudo-elliptical lensing potential
    of a Non-Singular Isothermal Sphere density profile lens.
    """

    def __init__(self,
                 total_items=1,
                 batch_size=1,
                 kappa_fov=7.68,
                 src_fov=3.,
                 pixels=128,
                 x_c=0.001,  # arcsec
                 z_source=2.,
                 z_lens=0.5,
                 norm=True,
                 model="raytracer",  # alternative is rim
                 method="analytic"
                 ):
        self.batch_size = batch_size
        self.total_items = total_items
        self.norm = norm
        self.model = model.lower()
        self.kappa_fov = kappa_fov
        self.src_fov = src_fov
        self.pixels = pixels
        self.method = method

        self.x_c = tf.constant(x_c, tf.float32)

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, pixels)
        x = tf.cast(x, tf.float32)
        self.x_source, self.y_source = [xx * src_fov/ 2 for xx in tf.meshgrid(x, x)]
        self.theta1, self.theta2 = [xx * kappa_fov / 2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_fov / 2
        self._physical_info(z_source, z_lens)
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
        return kappa, alpha  # (X, Y)

    def generate_batch_rim(self):
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-.5, maxval=.5)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-.5, maxval=.5)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=0.2)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        r_ein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1, maxval=2.)

        xs = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.1, maxval=0.1)
        ys = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.1, maxval=0.1)
        e = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=0.3)
        phi_s = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        w = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.1, maxval=0.2)

        kappa = self.kappa_field(xlens, ylens, elp, phi, r_ein)
        source = self.source_model(xs, ys, e, phi_s, w)
        lensed_image = self.lens_source(source, r_ein, elp, phi, xlens, ylens)
        return lensed_image, source, kappa  # (X, Y1, Y2)

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

    def lens_source_func(self, r_ein, e, phi, x0, y0, xs, ys, es, w):
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
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        return lens

    def src_coord_to_pix(self, x, y):
        dx = self.src_fov / (self.pixels - 1)
        xmin = -0.5 * self.src_fov
        ymin = -0.5 * self.src_fov
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def source_model(self, x0, y0, elp, phi, w):  # for rim, simple gaussian for testing the model
        beta1 = self.x_source - x0
        beta2 = self.y_source - y0
        _beta1 = beta1 * np.cos(phi) + beta2 * np.sin(phi)
        _beta2 = -beta1 * np.sin(phi) + beta2 * np.cos(phi)
        rho_sq = _beta1 ** 2 / (1 - elp) + _beta2 ** 2 * (1 - elp)
        source = np.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        return source[..., tf.newaxis]  # add channel dimension

    def kappa_field(self, xlens, ylens, elp, phi, r_ein):
        xk, yk = self.rotated_and_shifted_coords(xlens, ylens, phi)
        kappa = 0.5 * r_ein / (xk ** 2 / (1 - elp) + yk ** 2 * (1 - elp) + self.x_c ** 2) ** (1 / 2)
        return kappa[..., tf.newaxis]  # add channel dimension

    def approximate_deflection_angles(self, xlens, ylens, elp, phi, r_ein):
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(xlens, ylens, phi)
        denominator = (theta1 ** 2 / (1 - elp) + theta2 ** 2 * (1 - elp) + self.x_c ** 2) ** (1 / 2)
        alpha1 = r_ein * theta1 / (1 - elp) / denominator
        alpha2 = r_ein * theta2 * (1 - elp) / denominator
        # rotate back to original orientation of coordinate system
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return tf.stack([alpha1, alpha2], axis=-1)  # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    def analytical_deflection_angles(self, xlens, ylens, elp, phi, r_ein):
        b, q, s = self._param_conv(elp, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(xlens, ylens, -phi)
        psi = tf.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
        alpha1 = b / tf.sqrt(1. - q ** 2) * tf.math.atan(np.sqrt(1. - q ** 2) * theta1 / (psi + s))
        alpha2 = b / tf.sqrt(1. - q ** 2) * tf.math.atanh(np.sqrt(1. - q ** 2) * theta2 / (psi + s * q ** 2))
        # rotate back
        alpha1, alpha2 = self._rotate(alpha1, alpha2, phi)
        return tf.stack([alpha1, alpha2], axis=-1)

    def conv2d_deflection_angles(self, Kappa):
        alpha_x = tf.nn.conv2d(Kappa, self.Xconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap ** 2 / pi)
        alpha_y = tf.nn.conv2d(Kappa, self.Yconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap ** 2 / pi)
        return tf.concat([alpha_x, alpha_y], axis=-1)

    def rotated_and_shifted_coords(self, x0, y0, phi):
        """
        Important to shift then rotate, we move to the point of view of the
         lens before rotating the lens (rotation and translation are not commutative).
        """
        theta1 = self.theta1 - x0
        theta2 = self.theta2 - y0
        return self._rotate(theta1, theta2, phi)

    def _rotate(self, x, y, angle):
        return x * tf.math.cos(angle) + y * tf.math.sin(angle), -x * tf.math.sin(angle) + y * tf.math.cos(angle)

    def _physical_info(self, z_source, z_lens):
        self.Dls = COSMO.angular_diameter_distance_z1z2(z_lens, z_source).value  # value in Mpc
        self.Ds = COSMO.angular_diameter_distance(z_source).value
        self.Dl = COSMO.angular_diameter_distance(z_lens).value
        self.sigma_crit = (c ** 2 * self.Ds / (4 * np.pi * G * self.Dl * self.Dls) / (1e10 * M_sun) / u.Mpc).to(u.Mpc ** (-2)).value

    def set_deflection_angles_vars(self):
        self.kernel_side_l = 2 * self.pixels + 1  # this number should be odd
        self.cond = np.zeros((self.kernel_side_l, self.kernel_side_l))
        self.cond[self.pixels, self.pixels] = True
        self.dx_kap = self.kappa_fov / (self.pixels - 1)
        x = tf.linspace(-1., 1., self.kernel_side_l) * self.kappa_fov
        y = tf.linspace(-1., 1., self.kernel_side_l) * self.kappa_fov
        X_filt, Y_filt = tf.meshgrid(x, y)
        kernel_denom = tf.square(X_filt) + tf.square(Y_filt)
        Xconv_kernel = tf.divide(-X_filt, kernel_denom)
        B = tf.zeros_like(Xconv_kernel)
        Xconv_kernel = tf.where(self.cond, B, Xconv_kernel)
        Yconv_kernel = tf.divide(-Y_filt, kernel_denom)
        Yconv_kernel = tf.where(self.cond, B, Yconv_kernel)
        self.Xconv_kernel = tf.reshape(Xconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1, 1])
        self.Yconv_kernel = tf.reshape(Yconv_kernel, [self.kernel_side_l, self.kernel_side_l, 1, 1])
        x = tf.linspace(-1., 1.,
                        self.pixels) * self.src_fov / 2.  # TODO make options to have different size source image
        y = tf.linspace(-1., 1., self.pixels) * self.src_fov / 2.
        self.Xim, self.Yim = tf.meshgrid(x, y)
        self.Xim = tf.reshape(self.Xim, [-1, self.pixels, self.pixels, 1])
        self.Yim = tf.reshape(self.Yim, [-1, self.pixels, self.pixels, 1])

    def _param_conv(self, elp, r_ein):
        q = (1 - elp) / (1 + elp)  # axis ratio
        r_ein_conv = 2. * q * r_ein / tf.sqrt(1. + q ** 2)
        b = r_ein_conv * tf.sqrt((1 + q ** 2) / 2)
        s = self.x_c * tf.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, q, s
