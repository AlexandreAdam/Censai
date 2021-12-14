import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from censai.definitions import logit


class AnalyticalPhysicalModelv2:
    """
    This model produces convolved images of SIE+Shear with a Disc+Bulge Sersic Galaxy image
    """
    def __init__(
            self,
            pixels=128,
            image_fov=7.68,
            src_fov=3.0,
            psf_cutout_size=16,
            r_ein_min=0.5,
            r_ein_max=2.,
            n_min=1.,
            n_max=3.,
            r_eff_min=0.2,
            r_eff_max=1.,
            max_gamma=0.1,
            max_ellipticity=0.4,
            max_lens_shift=0.3,
            max_source_shift=0.3,
            noise_rms_min=0.001, # in percent of image peak
            noise_rms_max=0.02,
            noise_rms_mean=0.008,
            noise_rms_std=0.008,
            psf_fwhm_min=0.06,
            psf_fwhm_max=0.08,
            psf_fwhm_mean=0.07,
            psf_fwhm_std=0.01,
            sersic_i_eff=10.
    ):
        self.src_fov = src_fov
        self.pixels = pixels
        self.image_fov = image_fov
        self.s_scale = image_fov / pixels / 10000  # Make profile non-singular

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.image_fov/2
        xx, yy = np.meshgrid(x, x)
        # reshape for broadcast to [batch_size, pixels, pixels, channels]
        self.theta1 = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=tf.float32)
        self.theta2 = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

        # coordinates for psf
        self.r_squared = self.theta1**2 + self.theta2**2
        self.r_squared = tf.image.crop_to_bounding_box(self.r_squared,
                                                       offset_height=self.pixels//2 - psf_cutout_size//2,
                                                       offset_width=self.pixels//2 - psf_cutout_size//2,
                                                       target_width=psf_cutout_size,
                                                       target_height=psf_cutout_size)

        self.r_ein_min = r_ein_min
        self.r_ein_max = r_ein_max
        self.n_min = n_min
        self.n_max = n_max
        self.r_eff_max = r_eff_max
        self.r_eff_min = r_eff_min
        self.max_lens_shift = max_lens_shift
        self.max_source_shift = max_source_shift
        assert max_ellipticity <= 0.6, "Max ellipticity cannot go over 0.6"
        self.max_ellipticity = max_ellipticity
        self.max_gamma = max_gamma

        self.noise_rms_mean = noise_rms_mean
        self.noise_rms_std = noise_rms_std
        self.noise_rms_min = noise_rms_min
        self.noise_rms_max = noise_rms_max
        self.psf_fwhm_mean = psf_fwhm_mean
        self.psf_fwhm_std = psf_fwhm_std
        self.psf_fwhm_max = psf_fwhm_max
        self.psf_fwhm_min = psf_fwhm_min
        self.sersic_i_eff = sersic_i_eff

        self.noise_rms_pdf = tfp.distributions.TruncatedNormal(loc=noise_rms_mean, scale=noise_rms_std, low=noise_rms_min, high=noise_rms_max)
        self.psf_fwhm_pdf = tfp.distributions.TruncatedNormal(loc=psf_fwhm_mean, scale=psf_fwhm_std, low=psf_fwhm_min, high=psf_fwhm_max)

    def sersic_source(self, beta1, beta2, xs, ys, q, phi_s, n, r_eff):
        bn = 2 * n - 1/3  # approximate solution to gamma(2n;b_n) = 0.5 * Gamma(2n) for n > 0.36
        # shift and rotate coordinates to major/minor axis system
        beta1 = beta1 - xs
        beta2 = beta2 - ys
        beta1, beta2 = self._rotate(beta1, beta2, phi_s)
        r = tf.sqrt(beta1 ** 2 + beta2 ** 2 / q**2)
        return self.sersic_i_eff * tf.exp(-bn * (r / r_eff) ** (1 / n) - 1)

    def gaussian_source(self, beta1, beta2, xs, ys, q, phi_s, w):
        # shift and rotate coordinates to major/minor axis system
        beta1 = beta1 - xs
        beta2 = beta2 - ys
        beta1, beta2 = self._rotate(beta1, beta2, phi_s)
        rho_sq = beta1 ** 2 + beta2 ** 2 / q**2
        return tf.math.exp(-0.5 * rho_sq / w ** 2)

    # ================== Pixelated Source ====================
    def kappa_field(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.
    ):
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        if q > 0.95:
            return 0.5 * r_ein / tf.sqrt(theta1**2 + theta2**2/q**2 + self.s_scale**2)
        else:
            b, s = self._param_conv(q, r_ein)
            return 0.5 * b / tf.sqrt(theta1**2 + theta2**2/q**2 + s**2)

    def lens_source(
            self,
            source,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            psf_fwhm = 0.05
    ):
        if q > 0.95:
            alpha1, alpha2 = tf.split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        else:
            alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        warp = tf.concat([x_src_pix, y_src_pix], axis=3)
        im = tfa.image.resampler(source, warp)  # bilinear interpolation
        psf = self.psf_models(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_lens_source(
            self,
            source,
            noise_rms: float = 1e-3,
            psf_fwhm: float = 0.,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.05,
    ):
        im = self.lens_source(source, r_ein, q, phi, x0, y0, gamma_ext, phi_ext, psf_fwhm)
        peak = tf.reduce_max(im)
        im += tf.random.normal(shape=im.shape, mean=0., stddev=noise_rms * peak)
        return im

    # =============== Gaussian Source ========================
    def lens_source_gaussian_func(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0.,
            ys: float = 0.,
            qs: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        if q > 0.95:
            alpha1, alpha2 = tf.split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        else:
            alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        # sample intensity from functional form
        im = self.gaussian_source(beta1, beta2, xs, ys, qs, phi_s, w)
        psf = self.psf_models(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_lens_gaussian_source(
            self,
            noise_rms: float = 1e-3,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0,
            ys: float = 0,
            qs: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        im = self.lens_source_gaussian_func(r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, w, psf_fwhm)
        im += tf.random.normal(shape=im.shape, mean=0., stddev=np.atleast_1d(noise_rms)[:, None, None, None])
        return im

    # =============== Sersic Source ========================
    def lens_source_sersic_func(
            self,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0.,
            ys: float = 0.,
            qs: float = 1.,
            phi_s: float = 0.,
            n: float = 1,  # 1=exponential profile for disc, 4=Vaucouleur
            r_eff: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        if q > 0.95:
            alpha1, alpha2 = tf.split(self.approximate_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        else:
            alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        im = self.sersic_source(beta1, beta2, xs, ys, qs, phi_s, n, r_eff)
        psf = self.psf_models(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def lens_source_sersic_func_vec(self, x, psf_fwhm):
        # assume x has shape [batch_size, 13]
        r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, n, r_eff = [_x[:, None, None] for _x in tf.split(x, 13, axis=-1)]
        alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        im = self.sersic_source(beta1, beta2, xs, ys, qs, phi_s, n, r_eff)
        im = self.convolve_with_psf_vec(im, psf_fwhm)
        return im

    def noisy_lens_sersic_func(
            self,
            noise_rms: float = 1e-3,
            r_ein: float = 1.,
            q: float = 1.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0,
            ys: float = 0,
            qs: float = 1.,
            phi_s: float = 0.,
            n: float = 1,
            r_eff: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        im = self.lens_source_sersic_func(r_ein, q, phi, x0, y0, gamma_ext, phi_ext, xs, ys, qs, phi_s, n, r_eff, psf_fwhm)
        peaks = tf.reduce_max(im, axis=(1, 2, 3), keepdims=True)
        im += tf.random.normal(shape=im.shape, mean=0., stddev=np.atleast_1d(noise_rms)[:, None, None, None] * peaks)
        return im

    def noisy_lens_sersic_func_vec(self, x, noise_rms, psf_fwhm):
        im = self.lens_source_sersic_func_vec(x, psf_fwhm)
        peaks = tf.reduce_max(im, axis=(1, 2, 3))
        noise_rms = noise_rms * peaks
        im += tf.random.normal(shape=im.shape, mean=0., stddev=noise_rms)
        return im, noise_rms

    def lens_gaussian_source_func_given_alpha(
            self,
            alpha,
            xs: float = 0,
            ys: float = 0,
            q: float = 1.,
            phi_s: float = 0.,
            w: float = 0.1,
            psf_fwhm: float = 0.05
    ):
        alpha1, alpha2 = tf.split(alpha, 2, axis=-1)
        beta1 = self.theta1 - alpha1
        beta2 = self.theta2 - alpha2
        im = self.gaussian_source(beta1, beta2, xs, ys, q, phi_s, w)
        psf = self.psf_models(psf_fwhm)
        im = self.convolve_with_psf(im, psf)
        return im

    def src_coord_to_pix(self, x, y):
        dx = self.src_fov / (self.pixels - 1)
        xmin = -0.5 * self.src_fov
        ymin = -0.5 * self.src_fov
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def external_shear_potential(self, gamma_ext, phi_ext):
        rho = tf.sqrt(self.theta1**2 +self.theta2**2)
        varphi = tf.atan2(self.theta2**2, self.theta1**2)
        return 0.5 * gamma_ext * rho**2 * tf.cos(2 * (varphi - phi_ext))

    def external_shear_deflection(self, gamma_ext, phi_ext):
        # see Meneghetti Lecture Scripts equation 3.83 (constant shear equation)
        alpha1 = gamma_ext * (self.theta1 * tf.cos(phi_ext) + self.theta2 * tf.sin(phi_ext))
        alpha2 = gamma_ext * (-self.theta1 * tf.sin(phi_ext) + self.theta2 * tf.cos(phi_ext))
        return alpha1, alpha2

    def potential(self, r_ein, q, phi, x0, y0):  # arcsec^2
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        if q > 0.95:
            return r_ein * tf.sqrt(theta1**2 + theta2**2/q**2 + self.s_scale**2)
        else:
            b, s = self._param_conv(q, r_ein)
            phi_x, phi_y = tf.split(self.analytical_deflection_angles(r_ein, q, phi, x0, y0), 2, axis=-1)
            theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
            psi = tf.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
            varphi = theta1 * phi_x + theta2 * phi_y
            varphi -= b * q * s * tf.math.log(tf.sqrt(psi + s)**2 + (1 - q**2) * theta1**2)
            varphi += b * q * s * tf.math.log((1 + q) * s)
            return varphi

    def approximate_deflection_angles(self, r_ein, q, phi, x0, y0):
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        denominator = (theta1 ** 2 + theta2 ** 2 / q**2 + self.s_scale ** 2) ** (1 / 2)
        alpha1 = r_ein * theta1 / denominator
        alpha2 = r_ein * theta2 / q / denominator
        # # rotate back to original orientation of coordinate system
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return tf.concat([alpha1, alpha2], axis=-1)  # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    def analytical_deflection_angles(self, r_ein, q, phi, x0, y0):
        b, s = self._param_conv(q, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        psi = tf.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
        alpha1 = b * q / tf.sqrt(1. - q ** 2) * tf.math.atan(tf.sqrt(1. - q ** 2) * theta1 / (psi + s))
        alpha2 = b * q / tf.sqrt(1. - q ** 2) * tf.math.atanh(tf.sqrt(1. - q ** 2) * theta2 / (psi + s * q ** 2))
        # # rotate back
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return tf.concat([alpha1, alpha2], axis=-1)

    def rotated_and_shifted_coords(self, x0, y0, phi):
        ###
        # Important to shift then rotate, we move to the point of view of the
        # lens before rotating the lens (rotation and translation are not commutative).
        ###
        theta1 = self.theta1 - x0
        theta2 = self.theta2 - y0
        rho = tf.sqrt(theta1**2 + theta2**2)
        varphi = tf.atan2(theta2, theta1) - phi
        theta1 = rho * tf.cos(varphi)
        theta2 = rho * tf.sin(varphi)
        return theta1, theta2

    @staticmethod
    def _rotate(x, y, angle):
        return x * tf.cos(angle) + y * tf.sin(angle), -x * tf.sin(angle) + y * tf.cos(angle)

    def _param_conv(self, q, r_ein):
        r_ein_conv = 2. * q * r_ein / tf.sqrt(1. + q ** 2)
        b = r_ein_conv * tf.sqrt((1 + q ** 2) / 2)
        s = self.s_scale * tf.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, s

    @staticmethod
    def _qphi_to_ellipticity(q, phi):
        e1 = (1. - q) / (1. + q) * tf.cos(2 * phi)
        e2 = (1. - q) / (1. + q) * tf.sin(2 * phi)
        return e1, e2

    @staticmethod
    def _ellipticity_to_qphi(e1, e2):
        phi = tf.atan2(e2, e1)/2
        c = tf.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        return q, phi

    @staticmethod
    def _shear_polar_to_cartesian(r, phi):
        x = r * tf.cos(2 * phi)
        y = r * tf.sin(2 * phi)
        return x, y

    @staticmethod
    def _shear_cartesian_to_polar(x, y):
        r = tf.sqrt(x**2 + y**2)
        phi = tf.atan2(y, x)/2
        return r, phi

    def sigmoid_to_sersic_physical(self, x):
        # Method used to compute likelihood given model predictions
        r_ein, e1, e2, x0, y0, gamma1, gamma2, xs, ys, e1s, e2s, n, r_eff = tf.split(x, 13, axis=-1)
        r_ein = r_ein * (self.r_ein_max - self.r_ein_min) + self.r_ein_min
        e1 = self.max_ellipticity * (2 * e1 - 1)
        e2 = self.max_ellipticity * e2  # only allow positive values for e2, removes the angular degeneracy => angles are set btw 0 and pi
        q, phi = self._ellipticity_to_qphi(e1, e2)
        x0 = x0 * self.max_lens_shift - self.max_lens_shift
        y0 = y0 * self.max_lens_shift - self.max_lens_shift
        gamma1 = self.max_gamma * (2 * gamma1 - 1)
        gamma2 = self.max_gamma * gamma2
        gamma, gamma_phi = self._shear_cartesian_to_polar(gamma1, gamma2)
        xs = xs * self.max_source_shift - self.max_source_shift
        ys = ys * self.max_source_shift - self.max_source_shift
        e1s = self.max_ellipticity * (2 * e1s - 1)
        e2s = self.max_ellipticity * e2s
        qs, phi_s = self._ellipticity_to_qphi(e1s, e2s)
        n = n * (self.n_max - self.n_max) + self.n_min
        r_eff = r_eff * (self.r_eff_max - self.r_eff_min) + self.r_eff_min
        z = tf.concat([r_ein, q, phi, x0, y0, gamma, gamma_phi, xs, ys, qs, phi_s, n, r_eff], axis=1)
        return z

    def sersic_physical_to_logits(self, z):
        # method used to compute model loss in logit space
        r_ein, q, phi, x0, y0, gamma, gamma_phi, xs, ys, qs, phi_s, n, r_eff = tf.split(z, 13, axis=-1)
        r_ein = (r_ein - self.r_ein_min) / (self.r_ein_max - self.r_ein_min)
        e1, e2 = self._qphi_to_ellipticity(q, phi)
        e1 = (e1 / self.max_ellipticity + 1)/2
        e2 = e2 / self.max_ellipticity
        x0 = (x0 + self.max_lens_shift) / self.max_lens_shift
        y0 = (y0 + self.max_lens_shift) / self.max_lens_shift
        gamma1, gamma2 = self._shear_polar_to_cartesian(gamma, gamma_phi)
        gamma1 = (gamma1 / self.max_gamma + 1) / 2
        gamma2 = gamma2 / self.max_gamma
        xs = (xs + self.max_source_shift)/self.max_source_shift
        ys = (ys + self.max_source_shift)/self.max_source_shift
        e1s, e2s = self._qphi_to_ellipticity(qs, phi_s)
        e1s = (e1s / self.max_ellipticity + 1)/2
        e2s = e2s / self.max_ellipticity
        n = (n - self.n_min) / (self.n_max - self.n_min)
        r_eff = (r_eff - self.r_eff_min) / (self.r_eff_max - self.r_eff_min)
        x = logit(tf.concat([r_ein, e1, e2, x0, y0, gamma1, gamma2, xs, ys, e1s, e2s, n, r_eff], axis=-1))
        return x

    def psf_models_vec(self, psf_fwhm: tf.Tensor):
        psf_sigma = psf_fwhm[:, None, None, None] / (2 * tf.sqrt(2. * tf.math.log(2.)))
        psf = tf.math.exp(-0.5 * self.r_squared / psf_sigma**2)
        psf /= tf.reduce_sum(psf, axis=(1, 2, 3), keepdims=True)
        return psf

    def convolve_with_psf_vec(self, images, psf_fwhm: tf.Tensor):
        """
        Assume psf are images of shape [batch_size, pixels, pixels, channels]
        """
        psf = self.psf_models_vec(psf_fwhm)
        images = tf.transpose(images, perm=[3, 1, 2, 0])  # put batch size in place of channel dimension
        psf = tf.transpose(psf, perm=[1, 2, 0, 3])  # put different psf on "in channels" dimension
        convolved_images = tf.nn.depthwise_conv2d(images, psf, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
        convolved_images = tf.transpose(convolved_images, perm=[3, 1, 2, 0]) # put channels back to batch dimension
        return convolved_images

    def psf_models(self, psf_fwhm: float):
        psf_sigma =psf_fwhm/ (2 * tf.sqrt(2. * tf.math.log(2.)))
        psf = tf.math.exp(-0.5 * self.r_squared / psf_sigma**2)
        psf /= tf.reduce_sum(psf, axis=(1, 2, 3), keepdims=True)
        return psf

    def convolve_with_psf(self, images, psf):
        """
        Assume psf are images of shape [batch_size, pixels, pixels, channels]
        """
        images = tf.transpose(images, perm=[3, 1, 2, 0])  # put batch size in place of channel dimension
        psf = tf.transpose(psf, perm=[1, 2, 0, 3])  # put different psf on "in channels" dimension
        convolved_images = tf.nn.depthwise_conv2d(images, psf, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
        convolved_images = tf.transpose(convolved_images, perm=[3, 1, 2, 0]) # put channels back to batch dimension
        return convolved_images

    def draw_sersic_batch(self, batch_size):
        r_ein = tf.random.uniform(shape=(batch_size, 1), minval=self.r_ein_min, maxval=self.r_ein_max)
        e1 = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_ellipticity, maxval=self.max_ellipticity)
        e2 = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.max_ellipticity)
        q, phi = self._ellipticity_to_qphi(e1, e2)
        x0 = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_lens_shift, maxval=self.max_lens_shift)
        y0 = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_lens_shift, maxval=self.max_lens_shift)
        gamma1 = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_gamma, maxval=self.max_gamma)
        gamma2 = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.max_gamma)
        gamma, gamma_phi = self._shear_cartesian_to_polar(gamma1, gamma2)
        xs = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_source_shift, maxval=self.max_source_shift)
        ys = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_source_shift, maxval=self.max_source_shift)
        e1s = tf.random.uniform(shape=(batch_size, 1), minval=-self.max_ellipticity, maxval=self.max_ellipticity)
        e2s = tf.random.uniform(shape=(batch_size, 1), minval=0, maxval=self.max_ellipticity)
        qs, phi_s = self._ellipticity_to_qphi(e1s, e2s)
        n = tf.random.uniform(shape=(batch_size, 1), minval=self.n_min, maxval=self.n_max)
        r_eff = tf.random.uniform(shape=(batch_size, 1), minval=self.r_eff_min, maxval=self.r_eff_max)
        x = tf.concat([r_ein, q, phi, x0, y0, gamma, gamma_phi, xs, ys, qs, phi_s, n, r_eff], axis=1)

        noise_rms = self.noise_rms_pdf.sample(sample_shape=(batch_size))
        psf_fwhm = self.psf_fwhm_pdf.sample(sample_shape=(batch_size))

        lens, noise_rms = self.noisy_lens_sersic_func_vec(x, noise_rms, psf_fwhm)
        return lens, x, noise_rms, psf_fwhm


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    phys = AnalyticalPhysicalModelv2(
        pixels=128,
        image_fov=7.68,
        src_fov=3.0,
        psf_cutout_size=16,
        r_ein_min=0.5,
        r_ein_max=2.,
        n_min=1.,
        n_max=3.,
        r_eff_min=0.2,
        r_eff_max=1.,
        max_gamma=0.1,
        max_ellipticity=0.4,
        max_lens_shift=0.3,
        max_source_shift=0.3,
        noise_rms_min=0.001,
        noise_rms_max=0.05,
        noise_rms_mean=0.008,
        noise_rms_std=0.008,
        psf_fwhm_min=0.06,
        psf_fwhm_max=0.5,
        psf_fwhm_mean=0.1,
        psf_fwhm_std=0.1
    )
    for i in range(10):
        lens, x, noise_rms, psf_fwhm = phys.draw_sersic_batch(1)
        print(x)
        print(phys.sersic_physical_to_logits(x))
        print(lens.numpy().max())
        plt.imshow(lens[0, ..., 0], cmap="hot")
        plt.colorbar()
        plt.show()

