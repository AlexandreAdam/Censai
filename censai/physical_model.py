import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from censai.definitions import RayTracer


class PhysicalModel:
    """
    Physical model to be passed to RIM class at instantiation
    """
    def __init__(self, image_side=7.68, src_side=3.0, pixels=256, kappa_side=7.68, method="conv2d", checkpoint_path=None):
        self.image_side = image_side
        self.src_side = src_side
        self.pixels = pixels
        self.kappa_side = kappa_side
        self.method = method
        if metho == "unet":
            self.RT = RayTracer(trainable=False)
            self.RT.load_weights(checkpoint_path)
        self.set_deflection_angle_vars()

    def deflection_angle(self, kappa):
        if self.method == "conv2d":
            alpha_x = tf.nn.conv2d(kappa, self.xconv_kernel, [1, 1, 1, 1], "SAME")
            alpha_y = tf.nn.conv2d(kappa, self.yconv_kernel, [1, 1, 1, 1], "SAME")
            x_src = self.ximage - alpha_x
            y_src = self.yimage - alpha_y

        elif self.method == "unet":
            alpha = self.RT(kappa)
            alpha_x, alpha_y = tf.split(alpha, axis=3)

        else:
            raise ValueError(f"{self.method} is not in [conv2d, unet]")
        return x_src, y_src, alpha_x, alpha_y

    @staticmethod # TODO verify this has the correct form
    def log_likelihood(predictions, labels, noise_rms):
        return 0.5 * tf.reduce_mean((predictions - labels)**2/noise_rms**2)

    def forward(self, source, kappa, logkappa=True):
        if logkappa:
            kappa = 10**kappa
        x_src, y_src, _, _ = self.deflection_angle(kappa)
        im = self.lens_source(x_src, y_src, source)
        return im

    def noisy_forward(self, source, kappa, noise_rms, logkappa=True):
        im = self.forwad(source, kappa, logkappa)
        noise = tf.random.normal(im.shape, mean=0, stddev=noise_rms)
        return im + noise

    def lens_source(self, x_src, y_src, source):
        x_src_pix, y_src_pix = self.src_coord_to_pix(x_src, y_src)
        wrap = tf.concat([x_src_pix, y_src_pix], axis=3) # stack along channel dimension
        im = tfa.image.resampler(source, wrap) # bilinear interpolation of source on wrap grid
        return im

    def src_coord_to_pix(self, x, y):
        """
        Assume the coordinate center to be 0
        """
        dx = self.src_side/(self.pixels - 1)
        xmin = -0.5 * self.src_side
        ymin = -0.5 * self.src_side
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def set_deflection_angle_vars(self):
        dx_kap = self.kappa_side/(self.pixels - 1)

        # coordinate grid for kappa
        x = np.linspace(-1, 1, 2 * self.pixels + 1) # padding
        xx, yy = np.meshgrid(x, x)
        rho = xx**2 + yy**2
        xconv_kernel = -xx/rho * self.kappa_side
        yconv_kernel = -yy/rho * self.kappa_side
        xconv_kernel[self.pixels, self.pixels] = 0 # avoid exploding contribution of the center
        yconv_kernel[self.pixels, self.pixels] = 0
        # reshape to [filter_height, filter_width, in_channels, out_channels]
        self.xconv_kernel = tf.constant(xconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)
        self.yconv_kernel = tf.constant(yconv_kernel[..., np.newaxis, np.newaxis], dtype=tf.float32)

        # coordinates for image
        x = np.linspace(-1, 1, self.pixels) * self.image_side/2
        xx, yy = np.meshgrid(x, x) 
        # reshape for broadcast to [batch_size, pixels, pixels, 1]
        self.ximage = tf.constant(xx[np.newaxis, ..., np.newaxis], dtype=np.float32)
        self.yimage = tf.constant(yy[np.newaxis, ..., np.newaxis], dtype=tf.float32)

