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

    def jacobian(self, kappa):
        """
        We compute the jacobian in 3 steps:
            1) Solve the Poisson equation in Fourier space to find the potential
            2) Find the second derivatives of this potential (also in Fourier space)
            3) Take the inverse Fourier transform to find the Jacobian matrix
        """
        kappa = self.kappa_to_image_grid(kappa)
        batch_size = kappa.shape[0]
        pixels = self.pixels
        k = np.fft.fftfreq(n=2*pixels, d=pixels/self.image_fov)
        kx, ky = tf.cast(tf.meshgrid(k, k), tf.complex64)
        jacobians = []
        for i in tf.range(batch_size):
            kap = tf.image.pad_to_bounding_box(kappa[i, ...], offset_height=0, offset_width=0, target_width=2 * pixels, target_height=2 * pixels)
            kap = tf.signal.fft2d(tf.cast(kap[..., 0], tf.complex64)) / (2 / np.pi)**2
            phi = - tf.math.divide_no_nan(kap, kx**2 + ky**2)  # lensing potential
            phi_xx = tf.abs(tf.signal.ifft2d(- kx**2 * phi))[..., None]
            phi_yy = tf.abs(tf.signal.ifft2d(- ky**2 * phi))[..., None]
            phi_xy = tf.abs(tf.signal.ifft2d(- kx * ky * phi))[..., None]
            phi_xx = tf.image.crop_to_bounding_box(phi_xx,
                                                   offset_height=0,
                                                   offset_width=0,
                                                   target_width=self.pixels,
                                                   target_height=self.pixels)
            phi_xy = tf.image.crop_to_bounding_box(phi_xy,
                                                   offset_height=0,
                                                   offset_width=0,
                                                   target_width=self.pixels,
                                                   target_height=self.pixels)
            phi_yy = tf.image.crop_to_bounding_box(phi_yy,
                                                   offset_height=0,
                                                   offset_width=0,
                                                   target_width=self.pixels,
                                                   target_height=self.pixels)
            j1 = tf.concat([1 - phi_xx, -phi_xy], axis=-1)
            j2 = tf.concat([-phi_xy, 1 - phi_yy], axis=-1)
            jacobian = tf.stack([j1, j2], axis=-1)
            jacobians.append(jacobian)
        return tf.stack(jacobians, axis=0)

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
    phys = PhysicalModel(128)
    from censai import AnalyticalPhysicalModel
    # kappa = AnalyticalPhysicalModel(64).kappa_field(r_ein=np.array([1., 2.])[:, None, None, None])
    # psf = phys.psf_models(np.array([0.4, 0.12]))
    import matplotlib.pyplot as plt
    # x = tf.random.normal(shape=(2, 64, 64, 1))
    # y = phys.noisy_forward(x, kappa, np.array([0.01, 0.04]), psf)
    # # out = phys.convolve_with_psf(x, psf)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(y[0, ..., 0])
    # ax1.set_title("0.4")
    # ax2.imshow(y[1, ..., 0])
    # ax2.set_title("0.12")
    # # print(out.shape)
    # plt.show()
    # print(psf.numpy().sum(axis=(1, 2, 3)))
    kappa = AnalyticalPhysicalModel(128).kappa_field(r_ein=1.5, e=0.4)
    jacobian = phys.jacobian(kappa)
    jac_det = tf.linalg.det(jacobian)
    plt.imshow(jac_det[0], cmap="seismic", extent=[-7.69/2, 7.69/2]*2)
    plt.colorbar()
    contour = plt.contour(jac_det[0], levels=[0], cmap="gray", extent=[-7.69/2, 7.69/2]*2)
    plt.show()
