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
        # masks = tf.image.pad_to_bounding_box(masks, self.pixels//2, self.pixels//2, 2*self.pixels, 2*self.pixels)
        return masks

    def power_spectrum(self, x):
        mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        std = tf.reduce_mean((x - mean)**2, axis=(1, 2, 3), keepdims=True)
        x_padded = tf.image.pad_to_bounding_box((x - mean)/std, self.pixels//2, self.pixels//2, 2*self.pixels, 2*self.pixels)
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x_padded, tf.complex64)))
        x_hat = tf.image.crop_to_bounding_box(x_hat, offset_height=self.pixels, offset_width=self.pixels, target_width=self.pixels, target_height=self.pixels)
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            value = tf.reduce_sum(tf.abs(x_hat)**2 * self.masks[i][None, ...], axis=(1, 2, 3)) / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=value)
        ps = tf.transpose(ps.stack())  # reshape into [batch_size, bins]
        return ps

    def cross_correlation(self, x, y):
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        y_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(y, tf.complex64)))
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            value = tf.reduce_sum(tf.abs(tf.math.conj(x_hat) * y_hat) * self.masks[i][None, ...], axis=(1, 2, 3)) / (self.pixels//2)**2 / (4 * np.pi)
            value = value / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=value)
        ps = tf.transpose(ps.stack())  # reshape into [batch_size, bins]
        return ps

    def cross_correlation_coefficient(self, x, y):
        x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        y_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(y, tf.complex64)))
        Sxy = tf.math.conj(x_hat) * y_hat
        Sxx = tf.abs(x_hat)**2
        Syy = tf.abs(y_hat)**2
        gamma = tf.where(Sxx <= 1e-12, tf.zeros_like(Sxx), tf.abs(Sxy)**2 / Sxx)
        gamma = tf.where(Syy <= 1e-12, tf.zeros_like(Syy), gamma / Syy)
        ps = tf.TensorArray(DTYPE, size=self.bins)
        for i in range(self.bins):
            coherence = tf.reduce_sum(gamma * self.masks[i][None, ...], axis=(1, 2, 3))
            coherence = coherence / tf.reduce_sum(self.masks[i])
            ps = ps.write(index=i, value=coherence)
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

    def plot_cross_power_spectrum_statistics(self, x, y, fov, title=None, label=None, color="b"):
        assert len(x.shape) == 4
        amp = self.cross_correlation_coefficient(x, y)
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        sampling_rate = fov / self.pixels
        power = amp #/ sampling_rate
        power_mean = tf.reduce_mean(power, axis=0)
        power_std = tf.math.sqrt(tf.reduce_mean((power - power_mean[tf.newaxis, :])**2, axis=0))
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, power_mean, "-", color=color, label=label)
        ax.fill_between(f, power_mean - power_std, power_mean + power_std, color=color, alpha=0.2)
        ax.fill_between(f, power_mean - 2 * power_std, power_mean + 2 * power_std, color=color, alpha=0.1)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel(rf"$\gamma^2$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
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

    def plot_cross_power_spectrum(self, x, y, fov, title=None, label=None, color="b"):
        assert len(x.shape) == 4
        amp = self.cross_correlation_coefficient(x, y)[0]
        print(amp)
        _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
        f = (f[:-1] + f[1:])/2
        print(f)
        sampling_rate = fov / self.pixels
        power = amp #/ sampling_rate
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(f, power, "-", color=color, label=label)
        ax.set_xlabel(r"Frequency [arcsec$^{-1}$]")
        ax.set_ylabel(rf"$\gamma^2$")
        ax.set_xlim(f.min(), f.max())
        # ax.set_yscale("log")
        ax.set_title(title)
        return fig, ax

    # def plot_power_spectrum_histogram(self, x, fov, title=None, cmap="hot"):
    #     assert len(x.shape) == 4
    #     amp = self.power_spectrum(x)[0]
    #     _, f = np.histogram(np.arange(self.pixels//2)/fov, bins=self.bins)
    #     f = (f[:-1] + f[1:])/2
    #     power = amp / f


if __name__ == '__main__':
    ps = PowerSpectrum(bins=12, pixels=32)
    y = tf.random.normal(shape=[1, 32, 32, 1])
    # y = tf.nn.relu(y)
    # y /= tf.reduce_max(y, axis=(1, 2, 3), keepdims=True)
    _x = np.linspace(-1, 1, 32) * 10/2
    _x, _y = np.meshgrid(_x, _x)
    r = np.sqrt(_x**2 + _y**2)
    circle = np.cos(2 * np.pi * r / 3)
    x = 10 * tf.cast(circle[np.newaxis, ..., np.newaxis], tf.float32)
    circle2 = np.cos(2 * np.pi * r / 3)
    y = y + 10 * tf.cast(circle2[np.newaxis, ..., np.newaxis], tf.float32)

    # ell = ps.power_spectrum(x)

    x = tf.nn.relu(x)
    x /= tf.reduce_sum(x, axis=(1, 2, 3), keepdims=True)

    y = tf.nn.relu(y)
    y /= tf.reduce_sum(y, axis=(1, 2, 3), keepdims=True)

    # ps = PowerSpectrum(bins=5, pixels=16)
    # x = tf.random.normal(shape=[100, 16, 16, 1])
    # plt.imshow(tf.math.abs(tf.signal.fftshift(tf.signal.fft2d(tf.cast(x[0], tf.complex64)))))
    # plt.colorbar()
    # plt.show()
    # ell = ps.power_spectrum(x)
    # ps.plot_power_spectrum_statistics(x, 10)
    # ps.plot_power_spectrum(x, 10)
    # plt.show()


    # print(x.shape)
    # print(y.shape)
    # x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))
    # y_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(y, tf.complex64)))
    # Sxy = tf.math.conj(x_hat) * y_hat
    # Sxx = tf.abs(x_hat) ** 2
    # Syy = tf.abs(y_hat) ** 2
    # gamma = tf.where(tf.math.logical_or(Sxx == 0, Syy == 0), tf.zeros_like(Sxx), tf.abs(Sxy) ** 2 / (Sxx * Syy))
    #
    # print(tf.abs(x_hat))
    # plt.figure()
    # plt.imshow(tf.math.abs(x_hat)[0, ..., 0])
    # plt.figure()
    # plt.imshow(tf.math.abs(y_hat)[0, ..., 0])
    # plt.figure()
    # plt.imshow(gamma[0, ..., 0])
    # plt.colorbar()
    # plt.imshow(Sxx[0, ..., 0] / Sxx[0, 64, 64, 0])
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(Syy[0, ..., 0] / Syy[0, 64, 64, 0])
    # plt.colorbar()
    # plt.figure()
    # plt.imshow((tf.abs(Sxy)**2)[0, ..., 0])
    # print(tf.reduce_mean((tf.abs(Sxy)**2 / (Sxx * Syy + 1e-16))))
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow((tf.abs(Sxy)**2 / Sxx / Syy)[0, ..., 0])
    # plt.colorbar()
    # # ps.plot_power_spectrum_statistics(x, 10)
    # ps.plot_power_spectrum(x, 10)
    # plt.show()
    # ps.plot_cross_power_spectrum(x, y, 10)
    # plt.show()

    plt.figure()
    pixels = 32
    from scipy.signal import correlate2d
    x_std = np.random.normal(size=(pixels, pixels))
    # x_std /= x_std.sum()
    x_std = x[0, ..., 0].numpy()
    x_std = (x_std - x_std.mean()) / x_std.std()
    im = correlate2d(x_std, x_std, "same") #/ x_std.shape[0]**2 / (2 * np.pi)**2
    # im_f = np.fft.fftshift(np.fft.fft2(im))
    plt.imshow(im)
    plt.colorbar()

    import tensorflow as tf
    x = x_std[..., None]
    plt.figure()
    x_padded = tf.image.pad_to_bounding_box(x, pixels//2, pixels//2, 2 *pixels, 2 * pixels)
    plt.imshow(x_padded[..., 0])
    plt.figure()
    x_hat = tf.signal.fftshift(tf.signal.fft2d(tf.cast(x, tf.complex64)))
    # x_hat = tf.image.crop_to_bounding_box(x_hat, offset_height=pixels, offset_width=pixels, target_width=pixels, target_height=pixels)
    # P  = np.conj(x_hat) * x_hat
    P  = np.abs(x_hat)**2 / x_hat.shape[0]**2 / (2 * np.pi)**2
    plt.imshow(P)
    plt.colorbar()
    C = np.fft.ifft2(np.fft.fftshift(P))

    plt.figure()
    plt.imshow(np.abs(C))
    plt.colorbar()
    plt.show()
    # plt.figure()
    # x_hat = tf.signal.fft2d(tf.cast(x - tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True), tf.complex64))
    # correlation = tf.math.conj(x_hat) * x_hat# / 128**2
    # C = tf.signal.ifft2d(correlation)
    # plt.imshow(tf.abs(C)[0, ..., 0])
    # plt.colorbar()
    # plt.show()
