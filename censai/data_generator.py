import tensorflow as tf
import tensorflow_addons as tfa
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

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value, dtype=tf.float32) # value in Mpc
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value, dtype=tf.float32)


class NISGenerator(tf.keras.utils.Sequence):
    """
    Generate pairs of kappa field and deflection angles of pseudo-elliptical lensing potential 
    of a Non-Singular Isothermal Sphere density profile lens.
    """
    def __init__(self, 
            total_items=1, 
            batch_size=1, 
            kappa_side_length=7.68, 
            src_side_length=3., 
            pixels=128, 
            x_c=0.05, # arcsec
            z_source=2., 
            z_lens=0.5, 
            train=True, 
            norm=True,
            model="raytracer" # alternative is rim
            ):
        """
        :param kappa_side_length: Angular FOV of lens plane field (in mas)
        :param src_side_length: Angular FOV of source plane (in mas) -> should be much smaller than kappa FOV
        """
        self.batch_size = batch_size
        self.total_items = total_items
        self.train = train
        self.norm = norm
        self.model = model.lower()
        self.src_side = src_side_length
        self.pixels = pixels

        self.x_c = tf.constant(x_c, tf.float32)

        # instantiate coordinate grids
        x = tf.linspace(-1, 1, pixels)
        x = tf.cast(x, tf.float32)
        self.x_source, self.y_source = [xx * src_side_length/2 for xx in tf.meshgrid(x, x)]
        self.theta1, self.theta2 = [xx * kappa_side_length/2 for xx in tf.meshgrid(x, x)]
        self.dy_k = (x[1] - x[0]) * kappa_side_length/2
        self.angular_diameter_distances(z_source, z_lens)

    def __len__(self):
        return math.ceil(self.total_items / self.batch_size) 

    def __getitem__(self, idx):
        if self.model == "raytracer":
            return self.generate_batch()
        elif self.model == "rim":
            return self.generate_batch_rim()

    def generate_batch(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)
        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=0.2)
        phi = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        Rein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1, maxval=2.5)

        kappa = self.kappa_field(xlens, ylens, elp, phi, Rein)
        alpha = self.deflection_angles(xlens, ylens, elp, phi, Rein)
        return kappa, alpha #(X, Y)

    def generate_batch_rim(self):
        if self.train:
            tf.random.set_seed(None)
        else:
            tf.random.set_seed(42)

        xlens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        ylens = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-1, maxval=1)
        elp   = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0., maxval=0.2)
        phi   = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        r_ein = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=1, maxval=2.5)

        xs    = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-0.5, maxval=0.5)
        ys    = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=ylens-0.1, maxval=ylens+0.1)
        e     = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=0.3)
        phi_s = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=-pi, maxval=pi)
        w     = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0.01, maxval=0.2)

        kappa = self.kappa_field(xlens, ylens, elp, phi, r_ein)
        source = self.source_model(xs, ys, e, phi_s, w)
        lensed_image = self.lens_source(source, r_ein, elp, phi, xlens, ylens)
        return lensed_image, source, kappa #(X, Y1, Y2)

    def lens_source(self, source, r_ein, e, phi, x0, y0):
        theta1, theta2 = self.rotated_and_shifted_coords(phi, x0, y0)
        alpha = self.deflection_angles(x0, y0, e, phi, r_ein)
        beta1 = theta1 - alpha[..., 0]  # lens equation
        beta2 = theta2 - alpha[..., 1]
        x_src_pix, y_src_pix = self.src_coord_to_pix(beta1, beta2)
        wrap = tf.stack([x_src_pix, y_src_pix], axis=-1)
        im = tfa.image.resampler(source, wrap) # bilinear interpolation
        return im

    def src_coord_to_pix(self, x, y):
        dx = self.src_side/(self.pixels - 1)
        xmin = -0.5 * self.src_side
        ymin = -0.5 * self.src_side
        i_coord = (x - xmin) / dx
        j_coord = (y - ymin) / dx
        return i_coord, j_coord

    def source_model(self, x0, y0, elp, phi, w): # for rim, simple gaussian for testing the model
        beta1 = self.x_source - x0
        beta2 = self.y_source - y0
        _beta1 = beta1 * np.cos(phi) + beta2 * np.sin(phi)
        _beta2 = -beta1 * np.sin(phi) + beta2 * np.cos(phi) 
        rho_sq = _beta1**2/(1-elp) + _beta2**2 * (1 - elp)
        source = np.exp(-0.5 * rho_sq / w**2) / 2 / np.pi / w**2
        return source[..., tf.newaxis] # add channel dimension

    def kappa_field(self, xlens, ylens, elp, phi, r_ein):
        """
        :param xlens: Horizontal position of the lens  (in mas)
        :param ylens: Vertical position of the lens (in mas)
        :param x_c: Critical radius (in mas) where the density is flattened to avoid the singularity
        """
        xk, yk = self.rotated_and_shifted_coords(phi, xlens, ylens)
        kappa = 0.5 * r_ein / (xk**2/(1-elp) + yk**2*(1-elp) + self.x_c**2)**(1/2)
        return kappa[..., tf.newaxis] # add channel dimension

    def deflection_angles(self, xlens, ylens, elp, phi, r_ein):
        xk, yk = self.rotated_and_shifted_coords(phi, xlens, ylens)
        denominator = (xk**2/(1-elp) + yk**2*(1-elp) + self.x_c**2)**(1/2)
        alpha1 = r_ein * xk / (1 - elp) / denominator 
        alpha2 = r_ein * yk * (1 - elp) / denominator 
        return tf.stack([alpha1, alpha2], axis=-1) # stack alphas into tensor of shape [batch_size, pix, pix, 2]

    def rotated_and_shifted_coords(self, phi, x0, y0):
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

    def angular_diameter_distances(self, z_source, z_lens):
        self.Dls = tf.constant(cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value, dtype=tf.float32) # value in Mpc
        self.Ds = tf.constant(cosmo.angular_diameter_distance(z_source).value, dtype=tf.float32)
        self.Dl = tf.constant(cosmo.angular_diameter_distance(z_lens).value, dtype=tf.float32)

