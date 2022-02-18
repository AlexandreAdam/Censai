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
        masks = masks.stack()
        return masks

    def power_spectrum(self, x):
        flux = tf.reduce_sum(x, axis=(1, 2), keepdims=True)
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x/flux, tf.complex64)))
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            value = tf.reduce_sum(tf.abs(x_hat)**2 * self.masks[i][None, ...], axis=(1, 2)) / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=value)
        ps = tf.transpose(ps.stack())  # reshape into [batch_size, bins]
        return ps

    def cross_power_spectrum(self, x, y):
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x/tf.reduce_sum(x, axis=(1, 2), keepdims=True), tf.complex64)))
        y_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(y/tf.reduce_sum(y, axis=(1, 2), keepdims=True), tf.complex64)))
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            value = tf.reduce_sum(tf.abs(tf.math.conj(x_hat) * y_hat) * self.masks[i][None, ...], axis=(1, 2)) / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=value)
        ps = tf.transpose(ps.stack())  # reshape into [batch_size, bins]
        return ps

    def cross_correlation_coefficient(self, x, y):
        Pxy = self.cross_power_spectrum(x, y)
        Pxx = self.power_spectrum(x)
        Pyy = self.power_spectrum(y)
        gamma = Pxy / tf.sqrt(Pxx * Pyy + 1e-16)
        return gamma

    def plot_power_spectrum_statistics(self, x, fov, title=None, label=None, color="b"):
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
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_cross_power_spectrum_statistics(self, x, y, fov, title=None, label=None, color="b"):
        amp = self.cross_power_spectrum(x, y)
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
        ax.set_ylabel(rf"$\gamma$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_power_spectrum(self, x, fov, title=None, label=None, color="b"):
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
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_cross_power_spectrum(self, x, y, fov, title=None, label=None, color="b"):
        amp = self.cross_power_spectrum(x, y)[0]
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        sampling_rate = fov / self.pixels
        power = amp / sampling_rate
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, power, "-", color=color, label=label)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel(r"$P_{xy}$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_cross_correlation_coefficient(self, x, y, fov, title=None, label=None, color="b"):
        r = self.cross_correlation_coefficient(x, y)[0]
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        sampling_rate = fov / self.pixels
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, r, "-", color=color, label=label)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel(rf"$\gamma$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    def plot_cross_correlation_coefficient_statistics(self, x, y, fov, title=None, label=None, color="b"):
        r = self.cross_correlation_coefficient(x, y)
        r_mean = tf.reduce_mean(r, axis=0)
        r_std = tf.math.sqrt(tf.reduce_mean((r - r_mean[tf.newaxis, :])**2, axis=0))
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, r_mean, "-", color=color, label=label)
        ax.fill_between(f, r_mean - r_std, r_mean + r_std, color=color, alpha=0.2)
        ax.fill_between(f, r_mean - 2 * r_std, r_mean + 2 * r_std, color=color, alpha=0.1)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel(rf"$\gamma$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax
