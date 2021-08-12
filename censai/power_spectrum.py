import tensorflow as tf
import numpy as np
from censai.definitions import DTYPE
import matplotlib.pyplot as plt


class PowerSpectrum:
    def __init__(self, bins, pixels):
        assert bins < pixels//2
        self.pixels = pixels
        self.bins = bins
        self.masks = self.build_azimuthal_masks(bins)

    def build_azimuthal_masks(self, bins):
        x = tf.range(-self.pixels//2, self.pixels//2, dtype=DTYPE) + 0.5
        x, y = tf.meshgrid(x, x)
        r = tf.math.sqrt(x**2 + y**2)
        _, edges = np.histogram(np.arange(self.pixels//2), bins=bins)
        masks = tf.TensorArray(DTYPE, size=bins)  # equivalent to empty list and append, but using tensorflow
        for i in range(bins):
            mask = (r > edges[i]) & (r < edges[i + 1]) #& (x > 0)
            mask = tf.cast(mask, DTYPE)
            masks = masks.write(index=i, value=mask)
        masks = masks.stack()[..., tf.newaxis]
        return masks

    def power_spectrum(self, x):
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            value = tf.reduce_sum(tf.abs(x_hat)**2 * self.masks[i][None, ...], axis=(1, 2, 3)) / (self.pixels//2)**2 / (4 * np.pi)
            value = value / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=value)
        ps = tf.transpose(ps.stack())  # reshape into [batch_size, bins]
        return ps

    def plot_power_spectrum_statistics(self, x, fov, title=None, label=None, color="b"):
        assert len(x.shape) == 4
        amp = self.power_spectrum(x)
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        sampling_rate = fov / self.pixels
        power = amp / sampling_rate
        power_mean = tf.reduce_mean(power, axis=0)
        power_std = tf.math.sqrt(tf.reduce_mean((power - power_mean[tf.newaxis, :])**2, axis=0))
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, power_mean, "-", color=color, label=label)
        ax.fill_between(f, power_mean - power_std, power_mean + power_std, color=color, alpha=0.2)
        ax.fill_between(f, power_mean - 2 * power_std, power_mean + 2 * power_std, color=color, alpha=0.1)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel("Power")
        ax.set_xlim(f.min(), f.max())
        ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_power_spectrum(self, x, fov, title=None, label=None, color="b"):
        assert len(x.shape) == 4
        amp = self.power_spectrum(x)[0]
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        sampling_rate = fov / self.pixels
        power = amp / sampling_rate
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, power, "-", color=color, label=label)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel("Power")
        ax.set_xlim(f.min(), f.max())
        ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    # def plot_power_spectrum_histogram(self, x, fov, title=None, cmap="hot"):
    #     assert len(x.shape) == 4
    #     amp = self.power_spectrum(x)[0]
    #     _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
    #     f = (f[:-1] + f[1:])/2
    #     power = amp / f


if __name__ == '__main__':
    ps = PowerSpectrum(bins=32, pixels=128)
    x = tf.random.normal(shape=[100, 128, 128, 1])
    _x = np.linspace(-1, 1, 128) * 10/2
    _x, _y = np.meshgrid(_x, _x)
    r = np.sqrt(_x**2 + _y**2)
    circle = np.cos(2 * np.pi * r / 3)
    x = x + 10 * tf.cast(circle[np.newaxis, ..., np.newaxis], tf.float32)
    plt.imshow(x[0, ..., 0])
    plt.show()
    plt.imshow(tf.math.abs(tf.signal.fftshift(tf.signal.fft2d(tf.cast(x[0], tf.complex64)))))
    plt.colorbar()
    plt.show()
    ell = ps.power_spectrum(x)
    ps.plot_power_spectrum_statistics(x, 10)
    ps.plot_power_spectrum(x, 10)
    plt.show()
