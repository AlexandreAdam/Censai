import numpy as np
from censai.definitions import COSMO

# We keep this as reference of a previous version of this code


class SRC_KAPPA_Generator(object):
    def __init__(self, train_batch_size=1, test_batch_size=1, kap_side_length=7.68, src_side=3.0, num_src_side=48,
                 num_kappa_side=48):
        self.src_side = src_side
        self.kap_side_length = kap_side_length
        self.num_src_side = num_src_side
        self.num_kappa_side = num_kappa_side
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        x_s = np.linspace(-1., 1., self.num_src_side, dtype='float32') * self.src_side / 2
        y_s = np.linspace(-1., 1., self.num_src_side, dtype='float32') * self.src_side / 2
        self.dy_s = y_s[1] - y_s[0]
        self.Xsrc, self.Ysrc = np.meshgrid(x_s, y_s)
        x_k = np.linspace(-1., 1., self.num_kappa_side, dtype='float32') * self.kap_side_length / 2
        y_k = np.linspace(-1., 1., self.num_kappa_side, dtype='float32') * self.kap_side_length / 2
        self.dy_k = y_k[1] - y_k[0]
        self.Xkap, self.Ykap = np.meshgrid(x_k, y_k)
        self.Kappa_tr = np.zeros((train_batch_size, num_kappa_side, num_kappa_side, 1), dtype='float32')
        self.Source_tr = np.zeros((train_batch_size, num_src_side, num_src_side, 1), dtype='float32')
        self.Kappa_ts = np.zeros((test_batch_size, num_kappa_side, num_kappa_side, 1), dtype='float32')
        self.Source_ts = np.zeros((test_batch_size, num_src_side, num_src_side, 1), dtype='float32')

    def Kappa_fun(self, xlens, ylens, elp, phi, Rein, rc=0, Zlens=0.5, Zsource=2.0, c=299800000):
        Dds = COSMO.angular_diameter_distance_z1z2(Zlens, Zsource).value * 1e6
        Ds = COSMO.angular_diameter_distance(Zsource).value * 1e6
        sigma_v = np.sqrt(c ** 2 / (4 * np.pi) * Rein * np.pi / 180 / 3600 * Ds / Dds)
        A = self.dy_k / 2. * (2 * np.pi / (360 * 3600))
        rcord, thetacord = np.sqrt(self.Xkap ** 2 + self.Ykap ** 2), np.arctan2(self.Ykap, self.Xkap)
        thetacord = thetacord - phi
        Xkap, Ykap = rcord * np.cos(thetacord), rcord * np.sin(thetacord)
        rlens, thetalens = np.sqrt(xlens ** 2 + ylens ** 2), np.arctan2(ylens, xlens)
        thetalens = thetalens - phi
        xlens, ylens = rlens * np.cos(thetalens), rlens * np.sin(thetalens)
        r = np.sqrt((Xkap - xlens) ** 2 + ((Ykap - ylens) * (1 - elp)) ** 2) * (2 * np.pi / (360 * 3600))
        Rein = (4 * np.pi * sigma_v ** 2 / c ** 2) * Dds / Ds
        kappa = np.divide(np.sqrt(1 - elp) * Rein, (2 * np.sqrt(r ** 2 + rc ** 2)))
        mass_inside_00_pix = 2. * A * (np.log(2. ** (1. / 2.) + 1.) - np.log(2. ** (1. / 2.) * A - A) + np.log(
            3. * A + 2. * 2. ** (1. / 2.) * A))
        density_00_pix = np.sqrt(1. - elp) * Rein / (2.) * mass_inside_00_pix / ((2. * A) ** 2.)
        ind = np.argmin(r)
        kappa.flat[ind] = density_00_pix
        return kappa

    def gen_source(self, x_src=0, y_src=0, sigma_src=1, norm=False):
        Im = np.exp(-(((self.Xsrc - x_src) ** 2 + (self.Ysrc - y_src) ** 2) / (2. * sigma_src ** 2)))
        if norm is True:
            Im = Im / np.max(Im)
        return Im

    def draw_k_s(self, train_or_test):
        if (train_or_test == "train"):
            np.random.seed(seed=None)
            num_samples = self.train_batch_size
        if (train_or_test == "test"):
            np.random.seed(seed=136)
            num_samples = self.test_batch_size

        for i in range(num_samples):
            # parameters for kappa
            # np.random.seed(seed=155)
            xlens = np.random.uniform(low=-1.0, high=1.)
            ylens = np.random.uniform(low=-1.0, high=1.)
            elp = np.random.uniform(low=0.01, high=0.6)
            phi = np.random.uniform(low=0.0, high=2. * np.pi)
            Rein = np.random.uniform(low=2.0, high=4.0)

            # parameters for source
            sigma_src = np.random.uniform(low=0.5, high=1.0)
            x_src = np.random.uniform(low=-0.5, high=0.5)
            y_src = np.random.uniform(low=-0.5, high=0.5)
            norm_source = True

            if (train_or_test == "train"):
                self.Source_tr[i, :, :, 0] = self.gen_source(x_src=x_src, y_src=y_src, sigma_src=sigma_src,
                                                             norm=norm_source)
                self.Kappa_tr[i, :, :, 0] = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
            if (train_or_test == "test"):
                self.Source_ts[i, :, :, 0] = self.gen_source(x_src=x_src, y_src=y_src, sigma_src=sigma_src,
                                                             norm=norm_source)
                self.Kappa_ts[i, :, :, 0] = self.Kappa_fun(xlens, ylens, elp, phi, Rein)
        return

    def draw_average_k_s(self):
        src = self.gen_source(x_src=0., y_src=0., sigma_src=0.5, norm=True)
        kappa = self.Kappa_fun(0., 0., 0.02, 0., 3.0)
        src = src.reshape(1, self.num_src_side, self.num_src_side, 1)
        kappa = kappa.reshape(1, self.num_kappa_side, self.num_kappa_side, 1)

        src = np.repeat(src, self.train_batch_size, axis=0)
        kappa = np.repeat(kappa, self.train_batch_size, axis=0)
        return src, kappa