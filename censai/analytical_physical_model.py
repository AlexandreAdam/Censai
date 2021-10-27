import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


class AnalyticalPhysicalModel:
    def __init__(
            self,
            pixels=256,
            image_fov=7.68,
            src_fov=3.0,
            theta_c=1e-4):
        self.src_fov = src_fov
        self.pixels = pixels
        self.image_fov = image_fov
        self.theta_c = tf.constant(theta_c, dtype=tf.float32)

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.image_fov/2
        xx, yy = np.meshgrid(x, x)
        # reshape for broadcast to [batch_size, pixels, pixels, channels]
        self.theta1 = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=tf.float32)
        self.theta2 = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    def kappa_field(
            self,
            r_ein: float = 1.,
            e: float = 0.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.
    ):
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        return 0.5 * r_ein / tf.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)

    def lens_source(
            self,
            source,
            r_ein: float = 1.,
            e: float = 0.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.
    ):
        if e < 0.1:
            alpha1, alpha2 = tf.split(self.approximate_deflection_angles(r_ein, e, phi, x0, y0), 2, axis=-1)
        else:
            alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, e, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        warp = tf.stack([x_src_pix, y_src_pix], axis=4)
        im = tfa.image.resampler(source, warp) # bilinear interpolation
        return im

    def lens_source_func(
            self,
            r_ein: float = 1.,
            e: float = 0.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
            xs: float = 0.,
            ys: float = 0.,
            es: float = 0.,
            w: float = 0.1
    ):
        if e < 0.1:
            alpha1, alpha2 = tf.split(self.approximate_deflection_angles(r_ein, e, phi, x0, y0), 2, axis=-1)
        else:
            alpha1, alpha2 = tf.split(self.analytical_deflection_angles(r_ein, e, phi, x0, y0), 2, axis=-1)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        # lens equation
        beta1 = self.theta1 - alpha1 - alpha1_ext
        beta2 = self.theta2 - alpha2 - alpha2_ext
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        return lens

    def lens_source_func_given_alpha(
            self,
            alpha,
            xs: float = 0.,
            ys: float = 0.,
            es: float = 0.,
            w: float = 0.1
    ):
        alpha1, alpha2 = tf.split(alpha, 2, axis=-1)
        beta1 = self.theta1 - alpha1
        beta2 = self.theta2 - alpha2
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        return lens

    def noisy_lens_source(
            self,
            source,
            noise_rms: float = 1e-3,
            r_ein: float = 1.,
            e: float = 0.,
            phi: float = 0.,
            x0: float = 0.,
            y0: float = 0.,
            gamma_ext: float = 0.,
            phi_ext: float = 0.,
    ):
        im = self.lens_source(source, r_ein, e, phi, x0, y0, gamma_ext, phi_ext)
        im += tf.random.normal(shape=im.shape) * noise_rms
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

    def potential(self, r_ein, e, phi, x0, y0):  # arcsec^2
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        return r_ein * tf.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)

    def approximate_deflection_angles(self, r_ein, e, phi, x0, y0):
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, phi)
        denominator = (theta1 ** 2 / (1 - e) + theta2 ** 2 * (1 - e) + self.theta_c ** 2) ** (1 / 2)
        alpha1 = r_ein * theta1 / (1 - e) / denominator
        alpha2 = r_ein * theta2 * (1 - e) / denominator
        # rotate back to original orientation of coordinate system
        alpha1, alpha2 = self._rotate(alpha1, alpha2, -phi)
        return tf.concat([alpha1, alpha2], axis=-1)  # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    def analytical_deflection_angles(self, r_ein, e, phi, x0, y0):
        b, q, s = self._param_conv(e, r_ein)
        # rotate to major/minor axis coordinates
        theta1, theta2 = self.rotated_and_shifted_coords(x0, y0, -phi)
        psi = tf.sqrt(q ** 2 * (s ** 2 + theta1 ** 2) + theta2 ** 2)
        alpha1 = b / tf.sqrt(1. - q ** 2) * tf.math.atan(np.sqrt(1. - q ** 2) * theta1 / (psi + s))
        alpha2 = b / tf.sqrt(1. - q ** 2) * tf.math.atanh(np.sqrt(1. - q ** 2) * theta2 / (psi + s * q ** 2))
        # rotate back
        alpha1, alpha2 = self._rotate(alpha1, alpha2, phi)
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

    def _rotate(self, x, y, angle):
        return x * tf.math.cos(angle) + y * tf.math.sin(angle), -x * tf.math.sin(angle) + y * tf.math.cos(angle)

    def _param_conv(self, elp, r_ein):
        q = (1 - elp) / (1 + elp)  # axis ratio
        r_ein_conv = 2. * q * r_ein / tf.sqrt(1. + q ** 2)
        b = r_ein_conv * tf.sqrt((1 + q ** 2) / 2)
        s = self.theta_c * tf.sqrt((1 + q ** 2) / (2 * q ** 2))
        return b, q, s

