import tensorflow as tf
import numpy as np
from censai.models.ray_tracer import RayTracer
import tensorflow_addons as tfa
from censai.definitions import DTYPE


class PhysicalModel:
    """
    Physical model to be passed to RIM class at instantiation
    """
    def __init__(self, image_side=7.68, src_side=3.0, pixels=256, kappa_side=7.68, method="conv2d", noise_rms=1, logkappa=False, checkpoint_path=None):
        self.image_side = image_side
        self.src_side = src_side
        self.pixels = pixels
        self.kappa_side = kappa_side
        self.method = method
        self.noise_rms = noise_rms
        self.logkappa = logkappa
        if method == "unet":
            self.RT = RayTracer(trainable=False)
            self.RT.load_weights(checkpoint_path)
        self.set_deflection_angle_vars()

    def deflection_angle(self, kappa):
        if self.method == "conv2d":
            alpha_x = tf.nn.conv2d(kappa, self.xconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)
            alpha_y = tf.nn.conv2d(kappa, self.yconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)
            x_src = self.ximage - alpha_x
            y_src = self.yimage - alpha_y

        elif self.method == "unet":
            alpha = self.RT(kappa)
            alpha_x, alpha_y = tf.split(alpha, axis=3)
            x_src = self.ximage - alpha_x
            y_src = self.yimage - alpha_y
        else:
            raise ValueError(f"{self.method} is not in [conv2d, unet]")
        return x_src, y_src, alpha_x, alpha_y

    def log_likelihood(self, source, kappa, y_true):
        y_pred = self.forward(source, kappa)
        return 0.5 * tf.reduce_mean((y_pred - y_true)**2/self.noise_rms**2)

    def forward(self, source, kappa):
        if self.logkappa:
            kappa = 10**kappa
        im = self.lens_source(source, kappa)
        return im

    def noisy_forward(self, source, kappa, noise_rms):
        im = self.forward(source, kappa)
        noise = tf.random.normal(im.shape, mean=0, stddev=noise_rms)
        return im + noise

    def lens_source(self, source, kappa):
        x_src, y_src, _, _ = self.deflection_angle(kappa)
        x_src_pix, y_src_pix = self.src_coord_to_pix(x_src, y_src)
        wrap = tf.concat([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, wrap)  # bilinear interpolation of source on wrap grid
        return im

    def lens_source_and_compute_jacobian(self, source, kappa):
        """
        Note: this method will return a different picture than forward if image_side != kappa_side
        Args:
            source: Batch of source brightness distributions
            kappa: Batch of kappa maps

        Returns: lens image, jacobian matrix
        """
        # we have to compute everything here from scratch to get gradient paths
        x = tf.linspace(-1, 1, 2 * self.pixels + 1) * self.kappa_side
        x = tf.cast(x, DTYPE)
        theta_x, theta_y = tf.meshgrid(x, x)
        theta_x = theta_x[..., tf.newaxis, tf.newaxis]  # [..., in_channels, out_channels]
        theta_y = theta_y[..., tf.newaxis, tf.newaxis]
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(theta_x)
            tape.watch(theta_y)
            rho = theta_x**2 + theta_y**2
            kernel_x = - tf.math.divide_no_nan(theta_x, rho)
            kernel_y = - tf.math.divide_no_nan(theta_y, rho)
            # compute deflection angles
            alpha_x = tf.nn.conv2d(kappa, kernel_x, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)
            alpha_y = tf.nn.conv2d(kappa, kernel_y, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)
            # pad deflection angles with zeros outside of the scene (these are cropped out latter)
            alpha_x = tf.pad(alpha_x, [[0, 0]] + [[self.pixels//2, self.pixels//2 + 1]]*2 + [[0, 0]])
            alpha_y = tf.pad(alpha_y, [[0, 0]] + [[self.pixels//2, self.pixels//2 + 1]]*2 + [[0, 0]])
            # reshape thetas to broadcast properly onto alpha
            theta_x = tf.reshape(theta_x, [1, 2 * self.pixels + 1, 2 * self.pixels + 1, 1])
            theta_y = tf.reshape(theta_y, [1, 2 * self.pixels + 1, 2 * self.pixels + 1, 1])
            # lens equation
            x_src = theta_x - alpha_x
            y_src = theta_y - alpha_y
        # if target is not connected to source, make sure gradient return tensor of ZERO not NONE
        j11 = tape.gradient(x_src, theta_x, unconnected_gradients=tf.UnconnectedGradients.ZERO)#[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        j12 = tape.gradient(x_src, theta_y, unconnected_gradients=tf.UnconnectedGradients.ZERO)#[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        j21 = tape.gradient(y_src, theta_x, unconnected_gradients=tf.UnconnectedGradients.ZERO)#[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        j22 = tape.gradient(y_src, theta_y, unconnected_gradients=tf.UnconnectedGradients.ZERO)#[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        # put in a shape for which tf.linalg.det is easy to use (shape = [..., 2, 2])
        j1 = tf.concat([j11, j12], axis=3)
        j2 = tf.concat([j21, j22], axis=3)
        jacobian = tf.stack([j1, j2], axis=-1)
        # lens the source brightness distribution
        x_src = x_src[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        y_src = y_src[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        x_src_pix, y_src_pix = self.src_coord_to_pix(x_src, y_src)
        wrap = tf.concat([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, wrap)
        return im, jacobian

    def src_coord_to_pix(self, x, y):
        dx = self.src_side/(self.pixels - 1)
        xmin = -0.5 * self.src_side
        ymin = -0.5 * self.src_side
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def set_deflection_angle_vars(self):
        self.dx_kap = self.kappa_side/(self.pixels - 1)

        # coordinate grid for kappa
        x = np.linspace(-1, 1, 2 * self.pixels + 1) * self.kappa_side
        xx, yy = np.meshgrid(x, x)
        rho = xx**2 + yy**2
        xconv_kernel = -self._safe_divide(xx, rho)
        yconv_kernel = -self._safe_divide(yy, rho)
        # reshape to [filter_height, filter_width, in_channels, out_channels]
        self.xconv_kernel = tf.constant(xconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)
        self.yconv_kernel = tf.constant(yconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.image_side/2
        xx, yy = np.meshgrid(x, x) 
        # reshape for broadcast to [batch_size, pixels, pixels, 1]
        self.ximage = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=np.float32)
        self.yimage = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    @staticmethod
    def _safe_divide(num, denominator):
        out = np.zeros_like(num)
        out[denominator != 0] = num[denominator != 0] / denominator[denominator != 0]
        return out


class AnalyticalPhysicalModel: 
    def __init__(self, src_side=3.0, pixels=256, kappa_side=7.68, theta_c=0.1):
        self.src_side = src_side
        self.pixels = pixels
        self.kappa_side = kappa_side
        self.theta_c = tf.constant(theta_c, dtype=tf.float32)

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.kappa_side/2
        xx, yy = np.meshgrid(x, x) 
        # reshape for broadcast to [batch_size, pixels, pixels, channels]
        self.theta1 = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=tf.float32)
        self.theta2 = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    def kappa_field(self, r_ein, e, phi, x0, y0):
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        return 0.5 * r_ein / tf.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)

    def lens_source(self, source, r_ein, e, phi, x0, y0, gamma_ext, phi_ext):
        alpha1, alpha2 = self.deflection_angles(r_ein, e, phi, x0, y0)
        alpha1_ext, alpha2_ext = self.external_shear_deflection(gamma_ext, phi_ext)
        beta1 = self.theta1 - alpha1 - alpha1_ext # lens equation
        beta2 = self.theta2 - alpha2 - alpha2_ext
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        wrap = tf.stack([x_src_pix, y_src_pix], axis=4) 
        im = tfa.image.resampler(source, wrap) # bilinear interpolation 
        return im

    def noisy_lens_source(self, source, noise_rms, r_ein, e, phi, x0, y0, gamma_ext, phi_ext):
        im = self.lens_source(source, r_ein, e, phi, x0, y0, gamma_ext, phi_ext)
        im += tf.random.normal(shape=im.shape)
        return im

    def src_coord_to_pix(self, x, y):
        dx = self.src_side/(self.pixels - 1)
        xmin = -0.5 * self.src_side
        ymin = -0.5 * self.src_side
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def external_shear_potential(self, gamma_ext, phi_ext):
        rho = tf.sqrt(self.theta1**2 +self.theta2**2)
        varphi = tf.atan2(self.theta2**2 + self.theta1**2)
        return 0.5 * gamma_ext * rho**2 * tf.cos(2 * (varphi - phi_ext))

    def external_shear_deflection(self, gamma_ext, phi_ext):
        # see Meneghetti Lecture Scripts equation 3.83 (constant shear equation)
        alpha1 = gamma_ext * (self.theta1 * tf.cos(phi_ext) + self.theta2 * tf.sin(phi_ext))
        alpha2 = gamma_ext * (-self.theta1 * tf.sin(phi_ext) + self.theta2 * tf.cos(phi_ext))
        return alpha1, alpha2

    def rotated_and_shifted_coords(self, phi, x0, y0):
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

    def potential(self, r_ein, e, phi, x0, y0):  # arcsec^2
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        return r_ein * tf.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)

    def deflection_angles(self, r_ein, e, phi, x0, y0):  # arcsec
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        psi = tf.sqrt(theta1**2/(1-e) + (1-e)*theta2**2 + self.theta_c**2)
        alpha1 = r_ein * (theta1 / psi) / (1 - e)
        alpha2 = r_ein * (1 - e) * (theta2 / psi)
        return alpha1, alpha2

