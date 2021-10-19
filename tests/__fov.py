from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

p_fov = 0.07 * u.Mpc
h = cosmo.h
z = np.linspace(0.3, 1.5)
fov = h * p_fov / cosmo.angular_diameter_distance(z) * 180 * 3600 / np.pi

plt.plot(z, fov)
plt.show()
