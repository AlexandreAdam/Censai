import numpy as np
from sklearn.neighbors import NearestNeighbors
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import c, G, M_sun
from astropy import units as u
from astropy.io import fits
import os
import h5py
from argparse import ArgumentParser
from datetime import datetime
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from censai.utils import nullcontext
import time

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))


def exp1log_taylor(x):
    out = np.zeros_like(x)
    z = x[x <= 1]
    # out[x <= 1] = -np.euler_gamma + z - z ** 2 / 4 + z ** 3 / 18 - z ** 4 / 96 + z ** 5 / 600 - z ** 6 / 4320 # Taylor expansion
    out[x <= 1] = -0.57722 + 0.99999 * z - 0.24991 * z ** 2 + 0.05519 * z ** 3 - 0.00976 * z ** 4 + 0.00108 * z ** 5  # Allen and Hasting approximation
    z = x[(x > 1) & (x <= 10)]
    out[(x > 1) & (x <= 10)] = np.log(z) + np.exp(-z) / z * (0.26777 + 8.63476 * z + 18.05902 * z**2 + 8.57333 * z**3 + z**4) /\
                             (3.95850 + 21.09965 * z + 25.63296 * z**2 + 9.57332 * z**3 + z**4)  # Allen and Hastings approximation
    out[x > 10] = np.log(x[x > 10])
    return out


def tf_exp1log_taylor(x):
    """
    This is an approximation to Exp1(x) + log(x), where E1(x) = -Ei(-x), the exponential integral.

    For x < 10, we use the Allen and Hastings approximation to E1 (1955)
    For x > 10, E1(x) is negligible and can safely be ignored
    """
    out = tf.zeros_like(x)

    indices = tf.where(x <= 1)
    z = tf.gather_nd(x, indices)
    update = -0.57722 + 0.99999 * z - 0.24991 * z ** 2 + 0.05519 * z ** 3 -0.00976 *  z ** 4 + 0.00108 * z ** 5
    # update = -np.euler_gamma + z - z ** 2 / 4 + z ** 3 / 18 - z ** 4 / 96 + z ** 5 / 600 - z**6 / 4320  # Taylor expansion
    out = tf.tensor_scatter_nd_update(out, indices, update)

    indices = tf.where((x > 1) & (x <= 10))
    z = tf.gather_nd(x, indices)
    update = tf.math.log(z) + tf.exp(-z) / z\
                             * (0.26777 + 8.63476 * z + 18.05902 * z**2 + 8.57333 * z**3 + z**4)\
                             / (3.95850 + 21.09965 * z + 25.63296 * z**2 + 9.57332 * z**3 + z**4)
    out = tf.tensor_scatter_nd_update(out, indices, update)

    indices = tf.where(x > 10)
    z = tf.gather_nd(x, indices)
    update = tf.math.log(z)
    out = tf.tensor_scatter_nd_update(out, indices, update)
    return out


def numpy_dataset(coords, masses, ell_hat, batch_size):
    """
    Used in rasterize function when use_gpu=False
    Args:
        coords: Projected coordinate (2d) of particles
        masses: Masse of the particles
        ell_hat: Shape of the kernel
        batch_size: Number of particles data to output each iterations

    Yields: Batch of coords, masses and ell_hat
    """
    num_particle = coords.shape[0]
    iterations = list(range(0, int(num_particle / batch_size) * batch_size, batch_size)) + ["leftover"]
    for i in iterations:
        if i != "leftover":
            c = coords[i:i + batch_size, :][..., np.newaxis, np.newaxis, :]  # broadcast to shape of xi
            m = masses[i:i+batch_size, np.newaxis, np.newaxis]  # broadcast to shape of r_squared
            ell = ell_hat[i:i + batch_size, np.newaxis, np.newaxis]  # broadcast to shape of r_squared
            yield c, m, ell
        elif i == "leftover":
            leftover = num_particle % batch_size
            c = coords[-leftover:, :][..., np.newaxis, np.newaxis, :]  # broadcast to shape of xi
            m = masses[-leftover:, np.newaxis, np.newaxis]  # broadcast to shape of r_squared
            ell = ell_hat[-leftover:, np.newaxis, np.newaxis]  # broadcast to shape of r_squared
            yield c, m, ell


def tensorflow_generator(coords, masses, ell_hat):
    """
    Used in rasterize function when use_gpu=True
    Args:
        coords: Projected coordinate (2d) of particles
        masses: Masse of the particles
        ell_hat: Shape of the kernel

    Return: The callable generator to be fed to tensorflow dataset
    """
    num_particle = coords.shape[0]

    def generator():
        for i in range(num_particle):
            c = coords[i, :][np.newaxis, np.newaxis, :]
            m = masses[i, np.newaxis, np.newaxis]
            ell = ell_hat[i, np.newaxis, np.newaxis]
            yield c, m, ell
    return generator


def projection(proj):
    out = []
    for i in range(2):
        if proj[i] == "x":
            out.append(0)
        elif proj[i] == "y":
            out.append(1)
        elif proj[i] == "z":
            out.append(2)
        else:
            raise ValueError
    return out


def projection_is_done(done_subset, dim0, dim1):
    # check that the projection match (either [0, 1], [0, 2] or [1, 2])
    for i in range(done_subset.shape[0]):
        if done_subset[i, 1] == dim0 and done_subset[i, 2] == dim1:
            return True
    return False


def fixed_boundary_coordinates(coords, centroid, box_size):
    # account for the periodic box

    num_particles = coords.shape[0]

    new_coords = np.zeros(coords.shape)

    for i in range(3):
        CM_dist_0 = np.abs(coords[:, i] - centroid[i])
        CM_dist_1 = np.abs(coords[:, i] - np.ones(num_particles) * box_size - centroid[i])
        CM_dist_2 = np.abs(coords[:, i] + np.ones(num_particles) * box_size - centroid[i])
        # for each particle, we identify the coordinate closest to the center of mass, or center chosen
        particle_j = np.expand_dims(np.argmin(np.column_stack([CM_dist_0, CM_dist_1, CM_dist_2]), axis=1), axis=1)
        x_0 = coords[:, i]
        x_1 = coords[:, i] - np.ones(num_particles) * box_size
        x_2 = coords[:, i] + np.ones(num_particles) * box_size
        new_coords[:, i] = np.take_along_axis(np.column_stack([x_0, x_1, x_2]), particle_j, axis=1)[:, 0]
    return new_coords


def gaussian_kernel_rasterize(coords, mass, center, fov, dims=[0, 1], pixels=512, n_neighbors=64, fw_param=3, use_gpu=False, batch_size=1):
    """
    Rasterize particle cloud over the 2 dimensions, the output is a projected density
    """
    # Smooth particle mass over a region of size equal to the mean distance btw its n nearest neighbors in 3D
    print("Fitting Nearest Neighbors...")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    D = distances.max(axis=1)
    ell_hat = np.sqrt(103/1120) * D # Rau, S., Vegetti, S., & White, S. D. M. (2013). MNRAS, 430(3), 2232–2248. https://doi.org/10.1093/mnras/stt043

    # fixed fov scene
    xmin = center[0] - fov / 2
    xmax = center[0] + fov / 2
    ymin = center[1] - fov / 2
    ymax = center[1] + fov / 2
    if use_gpu:
        # The particle data can't be all loaded into the GPU,
        # so we use the tf.data.Dataset API to load a single batch at a time
        _np = tnp  # tensorflow version of numpy
        context = tf.device("/device:GPU:0")  # context is in GPU
        signature = (tf.TensorSpec(shape=(1, 1, 2), dtype=tf.float32),
                     tf.TensorSpec(shape=(1, 1), dtype=tf.float32),
                     tf.TensorSpec(shape=(1, 1), dtype=tf.float32))
        dataset = tf.data.Dataset.from_generator(tensorflow_generator(coords[:, dims], mass, ell_hat),
                                                 output_signature=signature)
        dataset = dataset.batch(batch_size, drop_remainder=False)#.prefetch(tf.data.experimental.AUTOTUNE)
        exp1_plus_log = tf_exp1log_taylor
    else:
        _np = np  # regular numpy
        context = nullcontext()  # no context needed
        dataset = numpy_dataset(coords[:, dims], mass, ell_hat, batch_size)
        exp1_plus_log = exp1log_taylor

    print("Rasterizing...")
    with context:
        x = _np.linspace(xmin, xmax, pixels, dtype=_np.float32)
        y = _np.linspace(ymin, ymax, pixels, dtype=_np.float32)
        x, y = _np.meshgrid(x, y)
        pixel_grid = _np.stack([x, y], axis=-1)

        Sigma = _np.zeros(shape=pixel_grid.shape[:2], dtype=_np.float32) # Convergence
        # Alpha = _np.zeros(shape=pixel_grid.shape, dtype=_np.float32)  # Deflection angles
        # Psi = _np.zeros_like(Sigma, dtype=_np.float32)  # Lensing potential
        # Gamma1 = _np.zeros_like(Sigma, dtype=_np.float32)  # Shear component 1
        # Gamma2 = _np.zeros_like(Sigma, dtype=_np.float32)  # Shear component 2
        # variance = _np.zeros_like(Sigma, dtype=_np.float32)
        # alpha_variance = _np.zeros_like(Alpha, dtype=_np.float32)
        for _coords, _mass, _ell_hat in dataset:
            xi = _coords - pixel_grid[_np.newaxis, ...]  # shape = [batch, pixels, pixels, xy]
            r_squared = xi[..., 0] ** 2 + xi[..., 1] ** 2  # shape = [batch, pixels, pixels]
            gaussian_fun = _np.exp(-0.5 * r_squared / _ell_hat ** 2)
            kappa_r = _mass * gaussian_fun / (2 * _np.pi * _ell_hat ** 2)
            Sigma += _np.sum(kappa_r, axis=0)
            ## Deflection angles
            # _alpha = _np.where(
            #     r_squared[..., None] > 0,
            #     _mass[..., None] / _np.pi * (gaussian_fun[..., None] - 1) * xi / r_squared[..., None],
            #     0.
            # )
            # Alpha += _np.sum(_alpha, axis=0)
            ## Lensing Potential
            # Psi += _np.sum(_mass / 2 / _np.pi * exp1_plus_log(r_squared / 2 / _ell_hat ** 2), axis=0)
            ## Shear
            # gamma_fun = r_squared + 2 * (1 - gaussian_fun) * _ell_hat**2
            # _gamma1 = _np.where(
            #     r_squared > 0,
            #     kappa_r * (xi[..., 0]**2 - xi[..., 1]**2) / r_squared**2 * gamma_fun,
            #     0.
            # )
            # Gamma1 += _np.sum(_gamma1, axis=0)
            # _gamma2 = _np.where(
            #     r_squared > 0,
            #     2 * kappa_r * xi[..., 0] * xi[..., 1] / r_squared ** 2 * gamma_fun,
            #     0.
            # )
            # Gamma2 += _np.sum(_gamma2, axis=0)
            ## Poisson shot noise of convergence field
            # variance += _np.sum((_mass * gaussian_fun / (2 * _np.pi * _ell_hat ** 2))**2, axis=0)
            ## Propagated uncertainty to deflection angles
            # A = gaussian_fun**2 - 2 * gaussian_fun
            # _alpha_variance = _np.where(
            #     r_squared[..., None]**2 > 0,
            #     (_mass[..., None] / _np.pi)**2 / r_squared[..., None]**2 * (A[..., None] + 1) * xi**2,
            #     0.
            # )
            # alpha_variance += _np.sum(_alpha_variance, axis=0)
    return Sigma#, Alpha, Psi, Gamma1, Gamma2, variance, alpha_variance


def load_halo(halo_id, particle_type, offsets, halo_offsets, snapshot_dir, snapshot_id):
    """
    We store here the logic to read from the file chunks of IllustrisTNG files. Another approach can be found
    here: https://github.com/illustristng/illustris_python
    Args:
        subhalo_id: The subhalo we wish to retrieve
        particle_type: 0: gas, 1: Dark matter, 4: Stars, 5: Black holes
        offsets: File offset matrix, with a row per chunk and column per particle type.
            Indicate the global particle ID of the first particle in a chunk.
        subhalo_offsets: Global particle ID of the first particle in the subhalo (for each type).
        snapshot_dir: Path of the directory where snapshot are stored
        snapshot_id: Id of the snapshot (e.g. 99 for the z=0 snapshot of TNG300-1)

    Returns: Subhalo particle coordinates and mass

    """
    assert particle_type in [0, 1, 4, 5]
    tot_chunks = offsets.shape[0]

    coords = []
    mass = []

    chunk = max(0, (halo_offsets[halo_id, particle_type] >= offsets[:, particle_type]).sum() - 1)  # first chunk
    start = halo_offsets[halo_id, particle_type] - offsets[chunk, particle_type]
    subhalo_length = halo_offsets[halo_id + 1, particle_type] - halo_offsets[halo_id, particle_type]
    if subhalo_length == 0:
        return None, None
    chunk_length = offsets[min(tot_chunks-1, chunk + 1), particle_type] - offsets[min(tot_chunks-2, chunk), particle_type]
    length = min(subhalo_length, chunk_length - start)
    remaining = subhalo_length - length
    i = 0
    while length > 0:
        snapshot_datapath = os.path.join(snapshot_dir, f"snap_{snapshot_id:03d}.{chunk + i}.hdf5")
        with h5py.File(snapshot_datapath, "r") as f:
            coords.append(f[f"PartType{particle_type:d}"]["Coordinates"][start:start + length, :] / 1e3)  # Mpc
            if particle_type != 1:
                mass.append(f[f"PartType{particle_type:d}"]["Masses"][start:start + length])
        i += 1
        start = 0
        chunk_length = offsets[min(tot_chunks-1, chunk + 1 + i), particle_type] - offsets[min(tot_chunks-2, chunk + i), particle_type]
        length = min(remaining, chunk_length)
        remaining -= length

    coords = np.concatenate(coords)
    if particle_type == 1:
        with h5py.File(snapshot_datapath, "r") as f:
            mass = dict(f['Header'].attrs.items())["MassTable"][1]  # get dm particle mass from simulation meta data
        mass = np.ones(coords.shape[0]) * mass
        return coords, mass
    mass = np.concatenate(mass)
    return coords, mass


def distributed_strategy(args):
    subhalo_ids = np.load(args.subhalo_id)
    if args.smoke_test:
        print("smoke_test")
        subhalo_ids = np.array([args.smoke_test_id])#np.array([52623])#np.array([41585])
    if "done.txt" in os.listdir(args.output_dir):  # for checkpointing
        done = np.loadtxt(os.path.join(args.output_dir, "done.txt"))
    else:
        done = np.array([])

    dims = projection(args.projection)

    zd = args.z_lens
    zs = args.z_source
    Ds = cosmo.angular_diameter_distance(zs)
    Dd = cosmo.angular_diameter_distance(zd)
    Dds = cosmo.angular_diameter_distance_z1z2(zd, zs)
    sigma_crit = (c ** 2 * Ds / (4 * np.pi * G * Dd * Dds) / (1e10 * M_sun)).to(u.Mpc ** (-2)).value

    with h5py.File(args.offsets, "r") as f:
        offsets = f["FileOffsets"]["SnapByType"][:]  # global particle number at the beginning of a chunk
        halo_offsets = f["Group"]["SnapByType"][:]
        subhalo_offsets = f["Subhalo"]["SnapByType"][:]
        subhalo_fileoffsets = f["FileOffsets"]["Subhalo"][:]

    tot_chunks = offsets.shape[0]
    with h5py.File(os.path.join(args.snapshot_dir, f"snap_{args.snapshot_id:03d}.0.hdf5"), "r") as f:
        box_size = dict(f['Header'].attrs.items())["BoxSize"]/1e3  # box size in Mpc

    # start hard work here
    for i in range(THIS_WORKER-1, subhalo_ids.size, N_WORKERS):
        subhalo_id = subhalo_ids[i]
        if subhalo_id in done:  # logic here is skipped if done.txt not in output_dir
            _done = done[done[:, 0] == subhalo_id]
            if len(_done.shape) == 1:
                _done = _done[np.newaxis, ...]
            if projection_is_done(done_subset=_done, dim0=dims[0], dim1=dims[1]):
                continue

        print(f"Started subhalo {subhalo_id} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")

        # Figure out which halo this subhalo belongs to, and get subhalo center of mass
        chunk = max(0, (subhalo_id > subhalo_fileoffsets).sum() - 1)  # first chunk
        subhalo_index = subhalo_id - subhalo_fileoffsets[chunk]
        chunk_length = subhalo_fileoffsets[min(tot_chunks-1, chunk + 1)] - subhalo_fileoffsets[min(tot_chunks-2, chunk)]
        while subhalo_index >= chunk_length:
            chunk += 1
            subhalo_index -= chunk_length
            chunk_length = subhalo_fileoffsets[min(tot_chunks-1, chunk + 1)] - subhalo_fileoffsets[min(tot_chunks-2, chunk)]
        fof_datapath = os.path.join(args.groupcat_dir, f"fof_subhalo_tab_{args.snapshot_id:03d}.{chunk}.hdf5")
        with h5py.File(fof_datapath, "r") as f:
            centroid = f["Subhalo"]["SubhaloCM"][subhalo_index] / 1e3
            halo_id = f["Subhalo"]["SubhaloGrNr"][subhalo_index]

        # load particles (0=gas, 1=DM, 4=stars, 5=black holes)
        coords = []
        mass = []
        _len = []
        for part_type in [0, 1, 4, 5]:
            print(f"Loading particle {part_type}...")
            coords_, mass_ = load_halo(halo_id, part_type, offsets, halo_offsets, args.snapshot_dir, args.snapshot_id)
            if coords_ is None:
                _len.append(0)
                continue
            _len.append(coords_.shape[0])
            coords.append(coords_)
            mass.append(mass_)
        coords = np.concatenate(coords)
        mass = np.concatenate(mass)
        print(f"Loaded {mass.shape[0]} particles in memory")

        # Adjust for periodic boundary conditions
        coords = fixed_boundary_coordinates(coords, centroid, box_size)
        x = coords[:, dims[0]]  # projection
        y = coords[:, dims[1]]

        #  figure out coordinate where kappa is at a maximum, this is so kappa maps are nicely centered
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=args.pixels, range=[[x.min(), x.max()], [y.min(), y.max()]],
                                                 weights=mass, density=False)  # global view of the subhalo
        cm_i, cm_j = np.unravel_index(np.argmax(heatmap), shape=heatmap.shape)
        cm_x = (xedges[cm_i] + xedges[cm_i + 1]) / 2  # find the position of the peak
        cm_y = (yedges[cm_j] + yedges[cm_j + 1]) / 2
        center = np.array([cm_x, cm_y])

        # select particles that will be in the cutout
        xmax = cm_x + args.fov/2
        xmin = cm_x - args.fov/2
        ymax = cm_y + args.fov/2
        ymin = cm_y - args.fov/2
        margin = 0.05 * args.fov   # allow for particle near the margin to be counted in
        select = (x < xmax + margin) & (x > xmin - margin) & (y < ymax + margin) & (y > ymin - margin)
        # kappa, alpha, psi, gamma1, gamma2, kappa_variance, alpha_variance = gaussian_kernel_rasterize(
        kappa = gaussian_kernel_rasterize(
            coords[select],
            mass[select],
            center,
            args.fov,
            dims=dims,
            pixels=args.pixels,
            n_neighbors=args.n_neighbors,
            fw_param=args.fw_param,
            use_gpu=args.use_gpu,
            batch_size=args.batch_size
        )
        kappa /= sigma_crit  # adimensional
        # psi /= sigma_crit    # Mpc^2/h^2
        # psi *= (1. / cosmo.angular_diameter_distance(args.z_lens).value * 3600 / np.pi * 180 * cosmo.h)**2  # convert to arcsec^2
        # alpha /= sigma_crit  # Mpc/h
        # alpha *= 1. / cosmo.angular_diameter_distance(args.z_lens).value * 3600 / np.pi * 180 * cosmo.h  # convert to arcsec
        # gamma1 /= sigma_crit  # adimensional
        # gamma2 /= sigma_crit  # adimensional
        # kappa_variance /= sigma_crit**2  # adimensional
        # alpha_variance /= sigma_crit**2  # Mpc^2/h^2
        # alpha_variance *= (1. / cosmo.angular_diameter_distance(args.z_lens).value * 3600 / np.pi * 180 * cosmo.h)**2  # convert to arcsec^2

        # create fits file and its header, then save result
        date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        print(f"Finished subhalo {subhalo_id} at {date}")

        header = fits.Header()
        header["SUBID"] = subhalo_id
        header["HALOID"] = halo_id
        header["MASS"] = (mass.sum(), "Total mass in the halo, in 10^{10} solar mass units")
        header["CUTMASS"] = (mass[select].sum(), "Total mass in the cutout in 10^{10} solar mass units")
        header["CREATED"] = date
        for part_type in [0, 1, 4, 5]:
            header[f"OFFSET{part_type:d}"] = subhalo_offsets[subhalo_id, part_type]
        header["FOV"] = (args.fov, "Field of view in Mpc")
        header["CD1_1"] = (((args.fov / args.pixels)*u.Mpc / cosmo.angular_diameter_distance(args.z_lens) * 180 / np.pi * 3600 * cosmo.h).value,
                           "Pixel scale in arcsec for x dimension")
        header["CD2_2"] = (((args.fov / args.pixels)*u.Mpc / cosmo.angular_diameter_distance(args.z_lens) * 180 / np.pi * 3600 * cosmo.h).value,
                           "Pixel scale in arcsec for y dimension")
        header["CD1_2"] = 0.
        header["CD2_1"] = 0.

        header["XDIM"] = dims[0]
        header["YDIM"] = dims[1]
        header["XCENTER"] = (center[0], "Mpc, Comoving coordinate in the simulation")
        header["YCENTER"] = (center[1], "Mpc, Comoving coordinate in the simulation")
        header["NPART"] = (coords.shape[0], "Total number of particles")
        title = ["GAS", "DM", "STARS", "BH"]
        for j, _ in enumerate([0, 1, 4, 5]):
            header[f"N{title[j]}"] = _len[j]
        header["NSELECT"] = (select.sum(), "Number of particles in the cutout")
        header["ZSOURCE"] = args.z_source
        header["ZLENS"] = args.z_lens
        header["NNEIGH"] = args.n_neighbors
        # header["FWPARAM"] = (args.fw_param, "FW at (1/x) maximum for smoothing")
        header["SIGCRIT"] = sigma_crit
        header["COSMO"] = "Planck18"
        hdu = fits.PrimaryHDU(kappa, header=header)
        # hdu1 = fits.ImageHDU(psi, name="Lensing potential")
        # hdu2 = fits.ImageHDU(alpha, name="Deflection Angles")
        # hdu3 = fits.ImageHDU(gamma1, name="Shear1")
        # hdu4 = fits.ImageHDU(gamma2, name="Shear2")
        # hdu5 = fits.ImageHDU(kappa_variance, name="Kappa Variance")
        # hdu6 = fits.ImageHDU(alpha_variance, name="Deflection Angles Variance")
        # hdul = fits.HDUList([hdu, hdu1, hdu2, hdu3, hdu4, hdu5, hdu6])
        hdul = fits.HDUList([hdu])
        hdul.writeto(os.path.join(args.output_dir, args.base_filenames + f"_{subhalo_id:06d}_{args.projection}.fits"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--offsets",            required=True,              help="Path to offset file (hfd5)")
    parser.add_argument("--groupcat_dir",       required=True,              help="Directory of of groupcat files for the snapshot of interest")
    parser.add_argument("--snapshot_dir",       required=True,              help="Root directory of the snapshot")
    parser.add_argument("--snapshot_id",        required=True, type=int,    help="Should match id of snapshot given in snapshot argument")
    parser.add_argument("--output_dir",         required=True, type=str,    help="Directory where to save raster images (fits file)")
    parser.add_argument("--subhalo_id",         required=True, type=str,    help="npy file that contains array of int32 index of subhalos to rasterize")
    parser.add_argument("--projection",         required=True, type=str,    help="2 characters, a combination of x, y and z (e.g. 'xy')")
    parser.add_argument("--base_filenames",     default="kappa")
    parser.add_argument("--pixels",             default=128,    type=int,   help="Number of pixels in the raster image")
    parser.add_argument("--n_neighbors",        default=64,     type=int,   help="Number of neighbors used to compute kernel length")
    parser.add_argument("--fw_param",           default=3,      type=float, help="Mean distance of neighbors is interpreted as "
                                                                                 "FW at (1/fw_param) of the maximum of the gaussian")
    parser.add_argument("--fov",                default=0.1,    type=float, help="Field of view of a scene in comoving Mpc")
    parser.add_argument("--z_source",           default=1.5, type=float)
    parser.add_argument("--z_lens",             default=0.5, type=float)
    parser.add_argument("--use_gpu",            action="store_true",        help="Will rasterize with tensorflow.experimental.numpy")
    parser.add_argument("--batch_size",         default=10, type=int,       help="Number of particles to rasterize at the same time")
    parser.add_argument("--smoke_test",         action="store_true")
    parser.add_argument("--smoke_test_id",      default=100, type=int,       help="Subhalo to do smoke test on")
    args = parser.parse_args()

    if THIS_WORKER > 1:
        time.sleep(5)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    distributed_strategy(args)
