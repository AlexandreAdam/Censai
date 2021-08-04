import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from censai.definitions import DTYPE
from censai.utils import nullcontext


class PhysicalModel:
    """
    Physical model to be passed to RIM class at instantiation
    """
    def __init__(
            self,
            pixels,            # 512
            psf_sigma=0.06,    # gaussian PSF
            src_pixels=None,   # 128 for cosmos
            kappa_pixels=None,
            image_fov=7.68,
            src_fov=3.0,
            kappa_fov=7.68,
            method="conv2d",
            noise_rms=1,
            raytracer=None
    ):
        if src_pixels is None:
            src_pixels = pixels  # assume src has the same shape
        if kappa_pixels is None:
            kappa_pixels = pixels
        self.image_fov = image_fov
        self.psf_sigma = psf_sigma
        self.src_fov = src_fov
        self.pixels = pixels
        self.src_pixels = src_pixels
        self.kappa_pixels = kappa_pixels
        self.kappa_fov = kappa_fov
        self.method = method
        self.noise_rms = noise_rms
        self.raytracer = raytracer
        self.set_deflection_angle_vars()
        self.PSF = self.psf_model()
        if kappa_pixels != pixels:
            self.kappa_to_image_grid = self._kappa_to_image_grid
        else:
            self.kappa_to_image_grid = tf.identity

    def deflection_angle(self, kappa):
        kappa = self.kappa_to_image_grid(kappa)  # resampling to shape of image
        if self.method == "conv2d":
            alpha_x = tf.nn.conv2d(kappa, self.xconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)
            alpha_y = tf.nn.conv2d(kappa, self.yconv_kernel, [1, 1, 1, 1], "SAME") * (self.dx_kap**2/np.pi)

        elif self.method == "unet":
            alpha = self.raytracer(kappa)
            alpha_x, alpha_y = tf.split(alpha, 2, axis=-1)

        elif self.method == "fft":
            """
            The convolution using the Convolution Theorem.
            Since we use FFT to justify this approach, we must zero pad the kernel and kappa map to transform 
            a 'circular convolution' (assumed by our use of FFT) into an an 'acyclic convolution' 
            (sum from m=0 to infinity).
            
            To do that, we pad our signal with N-1 trailing zeros for each dimension. N = 2*pixels+1 since 
            our kernel has this shape.
            
            This approach has complexity O((4*pixels)^2 * log^2(4 * pixels)), and is precise to about rms=2e-5 of the 
            true convolution for the deflection angles.
            """
            # pad the kernel and compute itf fourier transform
            xconv_kernel = tf.image.pad_to_bounding_box(self.xconv_kernel[..., 0], 0, 0, 4*self.pixels+1, 4*self.pixels+1)
            yconv_kernel = tf.image.pad_to_bounding_box(self.yconv_kernel[..., 0], 0, 0, 4*self.pixels+1, 4*self.pixels+1)
            x_kernel_tilde = tf.signal.fft2d(tf.cast(-xconv_kernel[..., 0], tf.complex64))
            y_kernel_tilde = tf.signal.fft2d(tf.cast(-yconv_kernel[..., 0], tf.complex64))

            batch_size = kappa.shape[0]
            alpha_x = []
            alpha_y = []
            for i in range(batch_size):
                kap = tf.image.pad_to_bounding_box(kappa[i, ...],  # pad kappa one by one to save memory space
                                                   offset_height=0,
                                                   offset_width=0,
                                                   target_width=4 * self.pixels + 1,
                                                   target_height=4 * self.pixels + 1)
                kappa_tilde = tf.signal.fft2d(tf.cast(kap[..., 0], tf.complex64))
                alpha_x.append(tf.math.real(tf.signal.ifft2d(kappa_tilde * x_kernel_tilde)) * (self.dx_kap**2/np.pi))
                alpha_y.append(tf.math.real(tf.signal.ifft2d(kappa_tilde * y_kernel_tilde)) * (self.dx_kap**2/np.pi))
            alpha_x = tf.stack(alpha_x, axis=0)[..., tf.newaxis]
            alpha_x = tf.image.crop_to_bounding_box(alpha_x,
                                                    offset_height=self.pixels,
                                                    offset_width=self.pixels,
                                                    target_width=self.pixels,
                                                    target_height=self.pixels)
            alpha_y = tf.stack(alpha_y, axis=0)[..., tf.newaxis]
            alpha_y = tf.image.crop_to_bounding_box(alpha_y,
                                                    offset_height=self.pixels,
                                                    offset_width=self.pixels,
                                                    target_width=self.pixels,
                                                    target_height=self.pixels)
        else:
            raise ValueError(f"{self.method} is not in [conv2d, unet, fft]")
        return alpha_x, alpha_y

    def log_likelihood(self, source, kappa, y_true):
        y_pred = self.forward(source, kappa)
        return 0.5 * tf.reduce_mean((y_pred - y_true)**2/self.noise_rms**2, axis=(1, 2, 3))

    def forward(self, source, kappa):
        im = self.lens_source(source, kappa)
        im = self.convolve_with_psf(im)
        return im

    def noisy_forward(self, source, kappa, noise_rms):
        im = self.lens_source(source, kappa)
        noise = tf.random.normal(im.shape, mean=0, stddev=noise_rms)
        out = self.convolve_with_psf(im)
        out = out + noise
        return out

    def lens_source(self, source, kappa):
        alpha_x, alpha_y = self.deflection_angle(kappa)
        x_src = self.ximage - alpha_x
        y_src = self.yimage - alpha_y
        x_src_pix, y_src_pix = self.src_coord_to_pix(x_src, y_src)
        warp = tf.concat([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, warp)  # bilinear interpolation of source on warp grid
        return im

    def lens_source_func(self, kappa, xs=0., ys=0., es=0., w=0.1):
        alpha_x, alpha_y = self.deflection_angle(kappa)
        # lens equation
        beta1 = self.ximage - alpha_x
        beta2 = self.yimage - alpha_y
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        lens = self.convolve_with_psf(lens)
        return lens

    def lens_source_func_given_alpha(self, alpha, xs=0., ys=0., es=0., w=0.1):
        alpha1, alpha2 = tf.split(alpha, 2, axis=-1)
        # lens equation
        beta1 = self.ximage - alpha1
        beta2 = self.yimage - alpha2
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        lens = self.convolve_with_psf(lens)
        return lens

    def lens_source_and_compute_jacobian(self, source, kappa):
        """
        Note: this method will return a different picture than forward if image_fov != kappa_fov
        Args:
            source: A source brightness distributions
            kappa: A kappa maps

        Returns: lens image, jacobian matrix
        """
        assert source.shape[0] == 1, "For now, this only works for a single example"
        # we have to compute everything here from scratch to get gradient paths
        x = tf.linspace(-1, 1, 2 * self.pixels + 1) * self.kappa_fov
        theta_x, theta_y = tf.meshgrid(x, x)
        theta_x = tf.cast(theta_x, DTYPE)
        theta_y = tf.cast(theta_y, DTYPE)
        theta_x = theta_x[..., tf.newaxis, tf.newaxis] # [..., in_channels, out_channels]
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
            # lens equation (reshape thetas to broadcast properly onto alpha)
            x_src = tf.reshape(theta_x, [1, 2 * self.pixels + 1, 2 * self.pixels + 1, 1]) - alpha_x
            y_src = tf.reshape(theta_y, [1, 2 * self.pixels + 1, 2 * self.pixels + 1, 1]) - alpha_y
        # Crop gradients
        j11 = tape.gradient(x_src, theta_x)[self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, ...]
        j12 = tape.gradient(x_src, theta_y)[self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, ...]
        j21 = tape.gradient(y_src, theta_x)[self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, ...]
        j22 = tape.gradient(y_src, theta_y)[self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, ...]
        # reshape gradients to [batch, pixels, pixels, channels] shape
        j11 = tf.reshape(j11, [1, self.pixels, self.pixels, 1])
        j12 = tf.reshape(j12, [1, self.pixels, self.pixels, 1])
        j21 = tf.reshape(j21, [1, self.pixels, self.pixels, 1])
        j22 = tf.reshape(j22, [1, self.pixels, self.pixels, 1])
        # put in a shape for which tf.linalg.det is easy to use (shape = [..., 2, 2])
        j1 = tf.concat([j11, j12], axis=3)
        j2 = tf.concat([j21, j22], axis=3)
        jacobian = tf.stack([j1, j2], axis=-1)
        # lens the source brightness distribution
        x_src = x_src[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        y_src = y_src[..., self.pixels//2: 3*self.pixels//2, self.pixels//2: 3*self.pixels//2, :]
        x_src_pix, y_src_pix = self.src_coord_to_pix(x_src, y_src)
        warp = tf.concat([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, warp)
        return im, jacobian

    def src_coord_to_pix(self, x, y):
        dx = self.src_fov / (self.src_pixels - 1)
        xmin = -0.5 * self.src_fov
        ymin = -0.5 * self.src_fov
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def kap_coord_to_pix(self, x, y):
        dx = self.kappa_fov / (self.kappa_pixels - 1)
        xmin = -0.5 * self.kappa_fov
        ymin = -0.5 * self.kappa_fov
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def _kappa_to_image_grid(self, kappa):
        batch_size = kappa.shape[0]
        xkappa = tf.cast(tf.linspace(-1, 1, self.pixels), dtype=DTYPE) * self.kappa_fov / 2
        xkappa, ykappa = tf.meshgrid(xkappa, xkappa)
        x_coord, y_coord = self.kap_coord_to_pix(xkappa[tf.newaxis, ..., tf.newaxis], ykappa[tf.newaxis, ..., tf.newaxis])
        warp = tf.concat([x_coord, y_coord], axis=-1)
        warp = tf.tile(warp, [batch_size, 1, 1, 1])  # make sure warp has same batch size has kappa
        kappa = tfa.image.resampler(kappa, warp)
        return kappa

    def set_deflection_angle_vars(self):
        self.dx_kap = self.kappa_fov / (self.pixels - 1)  # dx on image grid

        # Convolution kernel
        x = np.linspace(-1, 1, 2 * self.pixels + 1) * self.kappa_fov
        xx, yy = np.meshgrid(x, x)
        rho = xx**2 + yy**2
        xconv_kernel = -self._safe_divide(xx, rho)
        yconv_kernel = -self._safe_divide(yy, rho)
        # reshape to [filter_height, filter_width, in_channels, out_channels]
        self.xconv_kernel = tf.constant(xconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)
        self.yconv_kernel = tf.constant(yconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.image_fov / 2
        xx, yy = np.meshgrid(x, x) 
        # reshape for broadcast to [batch_size, pixels, pixels, 1]
        self.ximage = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=np.float32)
        self.yimage = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

    @staticmethod
    def _safe_divide(num, denominator):
        out = np.zeros_like(num)
        out[denominator != 0] = num[denominator != 0] / denominator[denominator != 0]
        return out

    def psf_model(self):
        pixel_scale = self.image_fov / self.pixels
        cutout_size = int(10 * self.psf_sigma / pixel_scale)
        r_squared = self.ximage**2 + self.yimage**2
        psf = np.exp(-0.5 * r_squared / self.psf_sigma**2)
        psf = tf.image.crop_to_bounding_box(psf,
                                            offset_height=self.pixels//2 - cutout_size//2,
                                            offset_width=self.pixels//2 - cutout_size//2,
                                            target_width=cutout_size,
                                            target_height=cutout_size)
        psf /= psf.numpy().sum()
        psf = tf.reshape(psf, shape=[cutout_size, cutout_size, 1, 1])
        return psf

    def convolve_with_psf(self, images):
        convolved_images = tf.nn.conv2d(images, self.PSF, [1, 1, 1, 1], padding="SAME")
        return convolved_images


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

