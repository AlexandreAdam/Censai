import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from censai.definitions import DTYPE


class PhysicalModel:
    """
    Physical model to be passed to RIM class at instantiation
    """
    def __init__(
            self,
            pixels,
            src_pixels=None,
            kappa_pixels=None,
            image_fov=7.68,
            src_fov=3.0,
            kappa_fov=7.68,
            method="conv2d",
            raytracer=None
    ):
        if src_pixels is None:
            src_pixels = pixels
        if kappa_pixels is None:
            kappa_pixels = pixels
        self.image_fov = image_fov
        self.src_fov = src_fov
        self.pixels = pixels
        self.src_pixels = src_pixels
        self.kappa_pixels = kappa_pixels
        self.kappa_fov = kappa_fov
        self.method = method
        self.raytracer = raytracer
        self.set_deflection_angle_vars()
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
            alpha_x = tf.TensorArray(dtype=DTYPE, size=batch_size)
            alpha_y = tf.TensorArray(dtype=DTYPE, size=batch_size)
            for i in tf.range(batch_size):
                kap = tf.image.pad_to_bounding_box(kappa[i, ...],  # pad kappa one by one to save memory space
                                                   offset_height=0,
                                                   offset_width=0,
                                                   target_width=4 * self.pixels + 1,
                                                   target_height=4 * self.pixels + 1)
                kappa_tilde = tf.signal.fft2d(tf.cast(kap[..., 0], tf.complex64))
                alpha_x = alpha_x.write(index=i, value=tf.math.real(tf.signal.ifft2d(kappa_tilde * x_kernel_tilde)) * (self.dx_kap**2/np.pi))
                alpha_y = alpha_y.write(index=i, value=tf.math.real(tf.signal.ifft2d(kappa_tilde * y_kernel_tilde)) * (self.dx_kap**2/np.pi))
            alpha_x = alpha_x.stack()[..., tf.newaxis]
            alpha_x = tf.image.crop_to_bounding_box(alpha_x,
                                                    offset_height=self.pixels,
                                                    offset_width=self.pixels,
                                                    target_width=self.pixels,
                                                    target_height=self.pixels)
            alpha_y = alpha_y.stack()[..., tf.newaxis]
            alpha_y = tf.image.crop_to_bounding_box(alpha_y,
                                                    offset_height=self.pixels,
                                                    offset_width=self.pixels,
                                                    target_width=self.pixels,
                                                    target_height=self.pixels)
        else:
            raise ValueError(f"{self.method} is not in [conv2d, unet, fft]")
        return alpha_x, alpha_y

    def log_likelihood(self, source, kappa, y_true, noise_rms, psf):
        y_pred = self.forward(source, kappa, psf)
        return 0.5 * tf.reduce_sum((y_pred - y_true) ** 2 / noise_rms[:, None, None, None] ** 2, axis=(1, 2, 3))

    @staticmethod
    def lagrange_multiplier(y_true, y_pred):
        return tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3), keepdims=True) / tf.reduce_sum(y_pred**2, axis=(1, 2, 3), keepdims=True)

    def forward(self, source, kappa, psf):
        im = self.lens_source(source, kappa)
        im = self.convolve_with_psf(im, psf)
        return im

    def noisy_forward(self, source, kappa, noise_rms, psf):
        im = self.lens_source(source, kappa)
        noise = tf.random.normal(im.shape, mean=0, stddev=np.atleast_1d(noise_rms)[:, None, None, None])
        out = self.convolve_with_psf(im, psf)  # convolve before adding noise, otherwise it correlates the noise
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

    def lens_source_func(self, kappa, xs=0., ys=0., es=0., w=0.1, psf_sigma=None):
        alpha_x, alpha_y = self.deflection_angle(kappa)
        # lens equation
        beta1 = self.ximage - alpha_x
        beta2 = self.yimage - alpha_y
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        psf_sigma = psf_sigma if psf_sigma is not None else self.image_fov / self.pixels
        psf = self.psf_models(psf_sigma)
        lens = self.convolve_with_psf(lens, psf)
        return lens

    def lens_source_func_given_alpha(self, alpha, xs=0., ys=0., es=0., w=0.1, psf_sigma=None):
        alpha1, alpha2 = tf.split(alpha, 2, axis=-1)
        # lens equation
        beta1 = self.ximage - alpha1
        beta2 = self.yimage - alpha2
        # sample intensity directly from the functional form
        rho_sq = (beta1 - xs) ** 2 / (1 - es) + (beta2 - ys) ** 2 * (1 - es)
        lens = tf.math.exp(-0.5 * rho_sq / w ** 2)  # / 2 / np.pi / w**2
        psf_sigma = psf_sigma if psf_sigma is not None else self.image_fov / self.pixels
        psf = self.psf_models(psf_sigma)
        lens = self.convolve_with_psf(lens, psf)
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
        kappa = self.kappa_to_image_grid(kappa)  # resampling to shape of image
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
        x_coord, y_coord = self.kap_coord_to_pix(self.xkappa, self.ykappa)
        warp = tf.concat([x_coord, y_coord], axis=-1)
        warp = tf.tile(warp, [batch_size, 1, 1, 1])  # make sure warp has same batch size has kappa
        kappa = tfa.image.resampler(kappa, warp)
        return kappa

    def set_deflection_angle_vars(self):
        self.dx_kap = self.kappa_fov / (self.pixels - 1)  # dx on image grid

        # Convolution kernel
        x = tf.cast(tf.linspace(-1, 1, 2 * self.pixels + 1), dtype=DTYPE) * self.kappa_fov
        xx, yy = tf.meshgrid(x, x)
        rho = xx**2 + yy**2
        xconv_kernel = -self._safe_divide(xx, rho)
        yconv_kernel = -self._safe_divide(yy, rho)
        # reshape to [filter_height, filter_width, in_channels, out_channels]
        self.xconv_kernel = tf.cast(xconv_kernel[..., tf.newaxis, tf.newaxis], dtype=DTYPE)
        self.yconv_kernel = tf.cast(yconv_kernel[..., tf.newaxis, tf.newaxis], dtype=DTYPE)

        # coordinates for image
        x = tf.cast(tf.linspace(-1, 1, self.pixels), dtype=DTYPE) * self.image_fov / 2
        xx, yy = tf.meshgrid(x, x)
        # reshape for broadcast to [batch_size, pixels, pixels, 1]
        self.ximage = tf.cast(xx[tf.newaxis, ..., tf.newaxis], dtype=DTYPE)
        self.yimage = tf.cast(yy[tf.newaxis, ..., tf.newaxis], dtype=DTYPE)

        # Coordinates for kappa
        xkappa = tf.cast(tf.linspace(-1, 1, self.pixels), dtype=DTYPE) * self.kappa_fov / 2
        xkappa, ykappa = tf.meshgrid(xkappa, xkappa)
        self.xkappa = tf.cast(xkappa[tf.newaxis, ..., tf.newaxis], dtype=DTYPE)
        self.ykappa = tf.cast(ykappa[tf.newaxis, ..., tf.newaxis], dtype=DTYPE)


    @staticmethod
    def _safe_divide(num, denominator):
        out = np.zeros_like(num)
        out[denominator != 0] = num[denominator != 0] / denominator[denominator != 0]
        return out

    def psf_models(self, psf_fwhm, cutout_size=16):
        psf_sigma = np.atleast_1d(psf_fwhm)[:, None, None, None] / (2 * np.sqrt(2 * np.log(2)))
        r_squared = self.ximage**2 + self.yimage**2
        psf = tf.math.exp(-0.5 * r_squared / psf_sigma**2)
        psf = tf.image.crop_to_bounding_box(psf,
                                            offset_height=self.pixels//2 - cutout_size//2,
                                            offset_width=self.pixels//2 - cutout_size//2,
                                            target_width=cutout_size,
                                            target_height=cutout_size)
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


if __name__ == '__main__':
    phys = PhysicalModel(64)
    from censai import AnalyticalPhysicalModel
    kappa = AnalyticalPhysicalModel(64).kappa_field(r_ein=np.array([1., 2.])[:, None, None, None])
    psf = phys.psf_models(np.array([0.4, 0.12]))
    import matplotlib.pyplot as plt
    x = tf.random.normal(shape=(2, 64, 64, 1))
    y = phys.noisy_forward(x, kappa, np.array([0.01, 0.04]), psf)
    # out = phys.convolve_with_psf(x, psf)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(y[0, ..., 0])
    ax1.set_title("0.4")
    ax2.imshow(y[1, ..., 0])
    ax2.set_title("0.12")
    # print(out.shape)
    plt.show()
    print(psf.numpy().sum(axis=(1, 2, 3)))
