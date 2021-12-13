import numpy as np
import tensorflow as tf
from scipy.special import exp1
from tqdm import tqdm
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import c, G, M_sun
import astropy.units as u


# # make a fake particle cloud
# pcloud = np.random.normal(size=(100, 2))/10
# mass = 1e4
#
# ell_hat = 0.01
# euler_gamma = np.euler_gamma
# pixels = 128
#
# zd = 0.5
# zs = 1.5
# Ds = cosmo.angular_diameter_distance(zs)
# Dd = cosmo.angular_diameter_distance(zd)
# Dds = cosmo.angular_diameter_distance_z1z2(zd, zs)
# sigma_crit = (c ** 2 * Ds / (4 * np.pi * G * Dd * Dds) / (1e10 * M_sun)).to(u.Mpc ** (-2)).value
#
# print("Rasterizing...")
# x = np.linspace(-1, 1, 128, dtype=np.float32)
# y = np.linspace(-1, 1, 128, dtype=np.float32)
# x, y = np.meshgrid(x, y)
# pixel_grid = np.stack([x, y], axis=-1)
#
# Sigma = np.zeros(shape=pixel_grid.shape[:2], dtype=np.float32) # Convergence
# Alpha = np.zeros(shape=pixel_grid.shape, dtype=np.float32)  # Deflection angles
# Psi = np.zeros(shape=pixel_grid.shape[:2], dtype=np.float32)  # Lensing potential
# variance = np.zeros(shape=pixel_grid.shape[:2], dtype=np.float32)
# alpha_variance = np.zeros(shape=pixel_grid.shape, dtype=np.float32)
# for _coords in tqdm(pcloud):
#     xi = _coords[None, None, None, :] - pixel_grid[np.newaxis, ...]  # shape = [batch, pixels, pixels, xy]
#     r_squared = xi[..., 0] ** 2 + xi[..., 1] ** 2  # shape = [batch, pixels, pixels]
#     Sigma += np.sum(mass * np.exp(-0.5 * r_squared / ell_hat ** 2) / (2 * np.pi * ell_hat ** 2), axis=0)
#     # Deflection angles
#     Alpha += np.sum(mass / np.pi * (np.exp(-0.5 * r_squared[..., None] / ell_hat ** 2) - 1) * xi / r_squared[..., None], axis=0)
#     # Lensing potential
#     r = np.sqrt(r_squared)
#     exp1_plus_log = tf.where(condition=(x==0), x=-0.5 * (euler_gamma + np.log(1/2/ell_hat**2)), y=0.5*exp1(r**2/2/ell_hat**2) + np.log(r))
#     Psi += np.sum(mass * ell_hat**2 / 2 / np.pi * exp1_plus_log, axis=0)
#     # Poisson shot noise of convergence field
#     variance += np.sum((mass * np.exp(-0.5 * r_squared / ell_hat ** 2) / (2 * np.pi * ell_hat ** 2))**2, axis=0)
#     # Propagated uncertainty to deflection angles
#     A = np.exp(-0.5 * r_squared / ell_hat ** 2)**2 - 2 * np.exp(-0.5 * r_squared / ell_hat ** 2)
#     _alpha_variance = tf.where(
#         condition=r_squared[..., None]**2 > 0,
#         x=(mass / np.pi)**2 / r_squared[..., None]**2 * (A[..., None] + 1) * xi**2,
#         y=0.
#     )
#     alpha_variance += np.sum(_alpha_variance, axis=0)


import matplotlib.pyplot as plt
# plt.imshow(Psi/sigma_crit)
# plt.colorbar()
# plt.show()
#
# plt.imshow(Sigma/sigma_crit)
# plt.colorbar()
# plt.show()

def exp1_plus_log(x):
    return np.where(x==0, -np.euler_gamma, exp1(x) + np.log(x))

def exp1log_taylor(x):
    out = np.zeros_like(x)
    out[x == 0] = -np.euler_gamma
    z = x[(x != 0) & (x <= 2)]
    # out[(x != 0) & (x <= 2)] = -np.euler_gamma + z - z ** 2 / 4 + z ** 3 / 18 - z ** 4 / 96 + z ** 5 / 600  # Taylor expansion
    out[(x != 0) & (x <= 2)] = -0.57722 + 0.99999 * z - 0.24991 * z ** 2 + 0.05519 * z ** 3 -0.00976 *  z ** 4 + 0.00108 * z ** 5  # Allen and Hasting approximation
    z = x[(x > 2) & (x <= 5)]
    out[(x > 2) & (x <= 5)] = np.log(z) + np.exp(-z) / z * (0.26777 + 8.63476 * z + 18.05902 * z**2 + 8.57333 * z**3) /\
                             (3.95850 + 21.09965 * z + 25.63296 * z**2 + 9.57332 * z**3)  # Allen and Hastings approximation
    out[x > 5] = np.log(x[x > 5])
    return out

def tf_exp1log_taylor(x):
    out = tf.zeros_like(x)
    # x = 0
    indices = tf.where(x == 0)
    out = tf.tensor_scatter_nd_update(out, indices, -tf.ones(indices.shape[0], dtype=tf.float32) * np.euler_gamma)
    # Allen and Hastings approximation
    # x < 1
    indices = tf.where((x != 0) & (x <= 2))
    z = tf.gather_nd(x, indices)
    # out = tf.tensor_scatter_nd_update(out, indices, -np.euler_gamma + z - z ** 2 / 4 + z ** 3 / 18 - z ** 4 / 96 + z ** 5 / 600) # Taylor expansion
    out = tf.tensor_scatter_nd_update(out, indices, -0.57722 + 0.99999 * z - 0.24991 * z ** 2 + 0.05519 * z ** 3 -0.00976 *  z ** 4 + 0.00108 * z ** 5)
    # 1 < x < 4
    indices = tf.where((x > 2) & (x <= 5))
    z = tf.gather_nd(x, indices)
    out = tf.tensor_scatter_nd_update(out, indices, tf.math.log(z) + tf.exp(-z) / z \
                                      * (0.26777 + 8.63476 * z + 18.05902 * z**2 + 8.57333 * z**3) \
                                      /(3.95850 + 21.09965 * z + 25.63296 * z**2 + 9.57332 * z**3))
    # x > 4 -> Ignore the Exponential Integral from here
    indices = tf.where(x > 5)
    z = tf.gather_nd(x, indices)
    out = tf.tensor_scatter_nd_update(out, indices, tf.math.log(z))
    return out

# x = np.linspace(0, 5, 1000)
# plt.plot(x, exp1_plus_log(x), color="k", label="exp1 + log")
# plt.plot(x, exp1log_taylor(x), color="b", label="log")
# plt.show()

x = np.linspace(0, 6, 1000).astype(np.float32)
plt.plot(x, (exp1_plus_log(x) - exp1log_taylor(x))/(exp1_plus_log(x) + 1e-16) * 100, color="k")
plt.plot(x, (exp1_plus_log(x) - tf_exp1log_taylor(x))/(exp1_plus_log(x) + 1e-16) * 100, color="r")

plt.show()