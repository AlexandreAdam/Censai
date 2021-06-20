import tensorflow as tf
import numpy as np
from censai.definitions import COSMO, DTYPE
from astropy.constants import G, c, M_sun
from astropy.io import fits
from astropy import units as u
from tensorflow_addons.image import rotate


class AugmentedTNGKappaGenerator:
    """
    This class contains the logic for data augmentation of kappa maps based
    on a rescaling factor, random shifts, crop and rotations.
    """
    def __init__(
            self,
            kappa_fits_files,               # fits files from rasterize scripts
            z_lens: float = None,           # new z_lens if provided
            z_source: float = None,
            crop: int = 0,
            rotate_by: str = "90",          # Either rotate by 90 degrees or 'uniform'
            min_theta_e: float = 1,
            max_theta_e: float = 5,
            rescaling_size: int = 100,
            rescaling_theta_bins: int = 10
    ):
        header = fits.open(kappa_fits_files[0])["PRIMARY"].header
        self.kappa_files = kappa_fits_files
        self.z_lens = z_lens if z_lens is not None else header["ZLENS"]
        self.z_source = z_source if z_source is not None else header["ZSOURCE"]
        self._crop = crop
        self.rescaling_size = rescaling_size
        self.rescaling_theta_bins = rescaling_theta_bins
        self.rotate_by = rotate_by

        # ====== Extract information common to all fits ======
        self.Dd = COSMO.angular_diameter_distance(self.z_lens)
        self.Ds = COSMO.angular_diameter_distance(self.z_source)
        self.Dds = COSMO.angular_diameter_distance_z1z2(self.z_lens, self.z_source)
        self.sigma_crit = (c ** 2 * self.Ds / (4 * np.pi * G * self.Dd * self.Dds)).to(u.kg * u.Mpc ** (-2))

        # Compute a rescaling factor given possibly new redshift pair
        self.sigma_crit_factor = (header["SIGCRIT"] * (1e10 * M_sun * u.Mpc ** (-2)) / self.sigma_crit).decompose().value

        pixels = fits.open(kappa_fits_files[0])["PRIMARY"].data.shape[0]  # pixels of the full cutout
        self.physical_pixel_scale = header["FOV"] / pixels * u.Mpc
        self.crop_pixels = pixels - 2 * crop  # pixels after crop
        self.pixel_scale = header["CD1_1"]  # pixel scale in arc seconds
        self.kappa_fov = self.pixel_scale * self.crop_pixels

        self.min_theta_e = min_theta_e
        self.max_theta_e = max_theta_e
        # ====================================================
        self.index = 0

    def crop(self, kappa):
        if len(kappa.shape) == 3:
            return kappa[self._crop: -self._crop, self._crop: -self._crop, ...]
        elif len(kappa.shape) == 4:
            return kappa[..., self._crop: -self._crop, self._crop: -self._crop, ...]

    def crop_and_shift(self, kappa):
        if len(kappa.shape) == 3:
            shift = np.random.randint(low=-self._crop + 1, high=self._crop - 1, size=2)
            return kappa[
                   self._crop + shift[0]: -(self._crop - shift[0]),
                   self._crop + shift[1]: -(self._crop - shift[1]), ...]
        elif len(kappa.shape) == 4:
            batch_size = kappa.shape[0]
            kap = []
            for j in range(batch_size):
                shift = np.random.randint(low=-self._crop + 1, high=self._crop - 1, size=2)
                kap.append(
                    kappa[j,
                    self._crop + shift[0]: -(self._crop - shift[0]),
                    self._crop + shift[1]: -(self._crop - shift[1]), ...]
                )
            kappa = tf.stack(kap, axis=0)
            return kappa

    def rotate(self, kappa):
        if len(kappa.shape) == 3:
            if self.rotate_by == "90":
                angle = np.random.randint(low=0, high=3, size=1)
                return tf.image.rot90(kappa, k=angle)
            elif self.rotate_by == "uniform":
                angle = np.random.uniform(low=-np.pi, high=np.pi, size=1)
                return rotate(kappa, angle, interpolation="nearest", fill_mode="constant")
        elif len(kappa.shape) == 4:
            batch_size = kappa.shape[0]
            if self.rotate_by == "90":
                rotated_kap = []
                for j in range(batch_size):
                        angle = np.random.randint(low=0, high=3, size=1)
                        rotated_kap.append(
                            tf.image.rot90(kappa, k=angle)
                        )
                return tf.stack(rotated_kap, axis=0)
            elif self.rotate_by == "uniform":
                angles = np.random.uniform(low=-np.pi, high=np.pi, size=batch_size)
                return rotate(kappa, angles, interpolation="nearest", fill_mode="constant")

    def einstein_radius(self, kappa):
        """
        Einstein radius is computed with the mass inside the Einstein ring, which corresponds to
        where kappa > 1.
        Args:
            kappa: A batch of kappa map of shape [batch_size, crop_pixels, crop_pixels, 1]

        Returns: Einstein radius in arcsecond of shape [batch_size]
        """
        mass_inside_einstein_radius = np.sum(kappa * (kappa > 1), axis=(1, 2, 3)) * self.sigma_crit * self.physical_pixel_scale ** 2
        return (np.sqrt(4 * G / c ** 2 * mass_inside_einstein_radius * self.Dds / self.Ds / self.Dd).decompose() * u.rad).to(u.arcsec).value

    def compute_rescaling_probabilities(self, kappa, rescaling_array):
        """
        Args:
            kappa: A single kappa map, of shape [crop_pixels, crop_pixels, 1]
            rescaling_array: An array of rescaling factor for which we need to compute the Einstein radius

        Returns: Probability of picking rescaling factor in rescaling array so that einstein radius has a
            uniform distribution between minimum and maximum allowed value (defined at instantiation of class)
        """
        p = np.zeros_like(rescaling_array)
        kappa = rescaling_array[..., np.newaxis, np.newaxis, np.newaxis] * kappa[np.newaxis, ...] # breadcast onto resaling array
        theta_e = self.einstein_radius(kappa)
        # compute theta distribution
        select = (theta_e >= self.min_theta_e) & (theta_e <= self.max_theta_e)
        if select.sum() == 0:  # no rescaling has landed in target range
            return p
        theta_hist, bin_edges = np.histogram(theta_e, bins=self.rescaling_theta_bins, range=[self.min_theta_e, self.max_theta_e], density=False)
        # for each theta_e, find bin index of our histogram. We give the left edges of the bin (param right=False)
        rescaling_bin = np.digitize(theta_e[select], bin_edges[:-1], right=False) - 1  # bin 0 is outside the range to the left by default
        theta_hist[theta_hist == 0] = 1  # give empty bins a weight
        p[select] = 1 / theta_hist[rescaling_bin]
        p /= p.sum()  # normalize our new probability distribution
        return p

    def rescale(self, kappa):
        """
        Draw a rescaling factor from Uniform(min_theta_e, max_theta_e)
        Args:
            kappa: A batch of kappa map of shape [batch_size, crop_pixels, crop_pixels, 1]

        Returns: kappa_rescaled, einstein_radius, rescaling_factors
        """
        batch_size = kappa.shape[0]
        kappa_rescaled = []
        rescaling_factors = []
        new_einstein_radius = []
        for j in range(batch_size):
            if kappa[j].max() <= 1:  # make sure at least a few pixels will be dense enough for deflection
                kappa[j] /= 0.95 * kappa[j].max()
            theta_e = self.einstein_radius(kappa[j][None, ...])[0]
            # Rough estimate of allowed rescaling factors
            rescaling_array = np.linspace(self.min_theta_e / theta_e, self.max_theta_e / theta_e, self.rescaling_size) * self.sigma_crit_factor
            rescaling_p = self.compute_rescaling_probabilities(kappa[j], rescaling_array)
            if rescaling_p.sum() == 0:
                rescaling = 1.
            else:
                rescaling = np.random.choice(rescaling_array, size=1, p=rescaling_p)[0]
            kappa_rescaled.append(rescaling * kappa[j])
            new_einstein_radius.append(self.einstein_radius(kappa_rescaled[-1][None, ...])[0])
            rescaling_factors.append(rescaling)
        kappa_rescaled = tf.stack(kappa_rescaled, axis=0)
        return kappa_rescaled, new_einstein_radius, rescaling_factors

    def draw_batch(self, batch_size, rescale: bool, shift: bool, rotate: bool, random_draw=True,
                   return_einstein_radius_init=False):
        if random_draw:
            batch_indices = np.random.choice(list(range(len(self.kappa_files))), replace=False, size=batch_size)
        else:
            batch_indices = list(range(self.index, min(self.index + batch_size, len(self.kappa_files))))
            if len(batch_indices) < batch_size:
                self.index = 0  # reset counter
            else:
                self.index += 1
        kappa = []
        kappa_ids = []
        for kap_index in batch_indices:
            kappa.append(fits.open(self.kappa_files[kap_index])["PRIMARY"].data[np.newaxis, ..., np.newaxis])
            kappa_ids.append(fits.open(self.kappa_files[kap_index])["PRIMARY"].header["SUBID"])
        kappa = tf.stack(kappa, axis=0)
        if rotate:
            kappa = self.rotate(kappa)
        if self._crop:
            if shift:
                kappa = self.crop_and_shift(kappa)
            else:
                kappa = self.crop(kappa)
        if rescale:
            theta_e_init = self.einstein_radius(kappa)
            kappa, theta_e, rescaling_factors = self.rescale(kappa)
            kappa = tf.cast(kappa, DTYPE)
            if return_einstein_radius_init:
                return kappa, theta_e, rescaling_factors, kappa_ids, theta_e_init
            else:
                return kappa, theta_e, rescaling_factors, kappa_ids
        else:
            theta_e = self.einstein_radius(kappa)
            rescaling_factors = np.ones(kappa.shape[0])
            kappa = tf.cast(kappa, DTYPE)
            return kappa, theta_e, rescaling_factors, kappa_ids

