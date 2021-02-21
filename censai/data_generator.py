import tensorflow as tf
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, c
from scipy.constants import pi
import astropy.units as u
import math
import numpy as np

c = tf.constant(c, dtype=tf.float32)
pi = tf.constant(pi, dtype=tf.float32)
deg2mas = pi / 180 / 3600
sqrt2 = tf.constant(tf.sqrt(2.), dtype=tf.float32)

# class GeneratorBase(tf.keras.utils.Sequence):

    # def __init__(self):

    # @abstractmethod
    # def generate_batch(self):
        # raise NotImplementedError


class Generator(tf.keras.utils.Sequence):
    """
    Class to generate source and kappa field during training and testing.
    #TODO check that this class wont work if kappa and source have different # of pixels
    """
    def __init__(self, total_items=1, batch_size=1, kappa_side_length=7.68, src_side_length=3., kappa_side_pixels=48, src_side_pixels=48, z_source=2., z_lens=0.5, train=True, norm=True):
        self.batch_size = batch_size
        self.total_items = total_items
        self.train = train
        self.norm = norm
        self.src_side_pixels = src_side_pixels

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, src_side_pixels)
        x = tf.cast(x, tf.float32)
        self.X_s, self.Y_s = [xx * src_side_length/2 for xx in tf.meshgrid(x, x)]
        self.X_k, self.Y_k = [xx * kappa_side_length/2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_side_length/2
        self.angular_diameter_distances(z_source, z_lens)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size) 

    def __getitem__(self, idx):
        return self.generate_batch()

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value * 1e6, dtype=tf.float32) # value in parsec
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value * 1e6, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value * 1e6, dtype=tf.float32)

    def generate_batch(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.01, maxval=0.6)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=2*pi)
        Rein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=2, maxval=4)

        #parameters for source
        sigma_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.5, maxval=1)
        x_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.5, maxval=0.5)
        y_src = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.5, maxval=0.5)

        kappa = self.kappa_field(xlens, ylens, elp, phi, Rein)
        source = self.gaussian_source(x_src, y_src, sigma_src)
        return source, kappa
        

    def kappa_field(self, xlens, ylens, elp, phi, Rein):
        """
        variables must have shape [batch_size, 1, 1] to be broadcastable
        """
        xk, yk = self.rotate(self.X_k, self.Y_k, phi)
        xs, ys = self.rotate(xlens, ylens, phi)
        A = self.dy_k/2*deg2mas
        r = tf.sqrt((xk - xs)**2 + ((yk - ys) * (1 - elp))**2 * deg2mas)
        # sigma_v = tf.sqrt(c**2/(4 * pi) * Rein * deg2mas * self.Ds / self.Dls) # velocity dispersion
        # Rein = 4 * pi * sigma_v**2 / c**2 * self.Dls / self.Ds  # Einstein radius
        kappa = tf.divide(tf.sqrt(1 - elp) * Rein, 2 * r)

        # normalize mass at the center
        mass_inside_center_pixel = 2 * A * (tf.math.log(sqrt2 + 1) - tf.math.log(sqrt2 * A - A) + tf.math.log(3 * A + 2*sqrt2 * A))
        density_center = tf.sqrt(1 - elp) * Rein/2 * mass_inside_center_pixel / (2 * A)**2
        density_center = tf.reshape(density_center, [-1]) # remove singleton dimensions for sparse update

        ind = tf.argmin(tf.reshape(r, [self.batch_size, -1]), axis=1)
        # broadcast index array to batch dim
        ind = [[i, ind[i]] for i in range(self.batch_size)]
        kappa = tf.reshape(kappa, [self.batch_size, -1])
        # sparse update of kappa
        density_center = tf.tensor_scatter_nd_add(kappa, ind, density_center)
        kappa = tf.reshape(kappa, r.shape)

        return kappa 

    @staticmethod
    @tf.function
    def rotate(x, y, phi):
        """
        Rotate the coordinate system by phi. 
        """
        rho = tf.sqrt(x**2 + y**2)
        theta = tf.math.atan2(y, x) - phi
        x_prime = rho * tf.math.cos(theta)
        y_prime = rho * tf.math.sin(theta)
        return x_prime, y_prime # shape [batch_size, pixel, pixel]


    def gaussian_source(self, x, y, sigma):
        rho_squared = (self.X_s - x)**2 + (self.Y_s - y)**2
        im = np.exp(-0.5 * rho_squared / sigma**2)
        if self.norm:
            im /= tf.reduce_max(im)
        return im


class NISGenerator(tf.keras.utils.Sequence):
    """
    Generate pairs of kappa field and deflection angles of pseudo-elliptical lensing potential 
    of a Non-Singular Isothermal Sphere density profile lens.
    """
    def __init__(self, total_items=1, batch_size=1, kappa_side_length=7.68, src_side_length=3., kappa_side_pixels=48, src_side_pixels=48, z_source=2., z_lens=0.5, train=True, norm=True):
        """
        :param kappa_side_length: Angular FOV of lens plane field (in mas)
        :param src_side_length: Angular FOV of source plane (in mas) -> should be much smaller than kappa FOV
        """
        self.batch_size = batch_size
        self.total_items = total_items
        self.train = train
        self.norm = norm
        self.src_side_pixels = src_side_pixels

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, src_side_pixels)
        x = tf.cast(x, tf.float32)
        self.X_s, self.Y_s = [xx * src_side_length/2 for xx in tf.meshgrid(x, x)]
        self.X_k, self.Y_k = [xx * kappa_side_length/2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_side_length/2
        self.angular_diameter_distances(z_source, z_lens)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size) 

    def __getitem__(self, idx):
        return self.generate_batch()

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value * 1e6, dtype=tf.float32) # value in parsec
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value * 1e6, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value * 1e6, dtype=tf.float32)

    def generate_batch(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.01, maxval=0.6)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=2*pi)
        Rein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=2, maxval=4)
        x_c = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1e-4, maxval=0.1)

        kappa = self.kappa_field(xlens, ylens, elp, phi, Rein, x_c)
        alpha = self.deflection_angles(xlens, ylens, elp, Rein, x_c)
        return kappa

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value * 1e6, dtype=tf.float32) # value in parsec
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value * 1e6, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value * 1e6, dtype=tf.float32)

    def kappa_field(self, xlens, ylens, elp, Rein, x_c):
        """
        :param xlens: Horizontal position of the lens  (in mas)
        :param ylens: Vertical position of the lens (in mas)
        :param x_c: Critical radius (in mas) where the density is flattened to avoid the singularity
        """
        xk, yk = self.rotate(self.X_k, self.Y_k, phi)
        xs, ys = self.rotate(xlens, ylens, phi)
        r = self.elliptical_radius(xk - xs, yk - ys, elp)
        return Rein * (r**2 + 2 * x_c**2) / 2 / (r**2 + x_c**2)**(3/2)

    def deflection_angles(self, xlens, ylens, elp, Rein, x_c):
        xk, yk = self.rotate(self.X_k, self.Y_k, phi)
        xs, ys = self.rotate(xlens, ylens, phi)
        r = self.elliptical_radius(xk - xs, yk - ys, elp)
        denominator = tf.sqrt(r**2 + x_c**2)
        alpha1 = Rein * xk / denominator
        alpha2 = Rein * yk / denominator
        return tf.stack([alpha1, alpha2], axis=-1) # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    @staticmethod
    @tf.function
    def rotate(x, y, phi):
        """
        Rotate the coordinate system by phi. 
        """
        rho = tf.sqrt(x**2 + y**2)
        theta = tf.math.atan2(y, x) - phi
        x_prime = rho * tf.math.cos(theta)
        y_prime = rho * tf.math.sin(theta)
        return x_prime, y_prime # shape [batch_size, pixel, pixel]

    @staticmethod
    @tf.function
    def elliptical_radius(x, y, e):
        return tf.sqrt(x**2 / (1 - e) + y**2 * (1-e))

