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

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
this_worker = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))


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


def gaussian_kernel_rasterize(coords, mass, center, fov, dims=[0, 1], pixels=512, n_neighbors=10):
    """
    Rasterize particle cloud over the 2 dimensions, the output is a projected density
    """
    # Smooth particle mass over a region of size equal to the mean distance btw its n nearest neighbors in 3D
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    distances = distances[:, 1:]  # the first column is 0 since the nearest neighbor of each point is the point itself, at a distance of zero.
    D = distances.mean(axis=1)  # characteristic distance used in kernel, correspond to FWHM of the kernel
    ell_hat = D / 2 / np.sqrt(2 * np.log(2))

    # ell_hat = D * np.sqrt(103 / 1120)  is used by Rau, S., Vegetti, S., & White, S. D. M. (2013). MNRAS, 430(3), 2232–2248. https://doi.org/10.1093/mnras/stt043
    # this corresponds D=(FW at 2/3 maximum), so the kernel is very sharp. We use FWHM so the kernel are more large but this drowns better the particle noise

    # pixel grid encompass the full scene, but variable pixel size depending on the scene
    #     xmin = coord[:, dims[0]].min()
    #     xmax = coord[:, dims[0]].max()
    #     ymin = coord[:, dims[1]].min()
    #     ymax = coord[:, dims[1]].max()
    #     x = np.linspace(xmin, xmax, pixels)
    #     y = np.linspace(ymin, ymax, pixels)
    #     x, y = np.meshgrid(x, y)

    xmin = center[0] - fov / 2
    xmax = center[0] + fov / 2
    ymin = center[1] - fov / 2
    ymax = center[1] + fov / 2
    x = np.linspace(xmin, xmax, pixels)
    y = np.linspace(ymin, ymax, pixels)
    x, y = np.meshgrid(x, y)
    pixel_grid = np.stack([x, y], axis=-1)

    num_particle = coords.shape[0]
    Sigma = np.zeros(shape=pixel_grid.shape[:2], dtype=np.float32)
    for i in range(num_particle):
        xi = coords[i, dims][np.newaxis, np.newaxis, :] - pixel_grid
        r_squared = xi[..., 0]**2 + xi[..., 1]**2
        Sigma += mass[i] * np.exp(-0.5 * r_squared / ell_hat[i] ** 2) / (2 * np.pi * ell_hat[i] ** 2)  # gaussian kernel
    return Sigma


def load_subhalo(subhalo_id, particle_type, offsets, subhalo_offsets, snapshot_dir, snapshot_id):

    coords = []
    mass = []

    chunk = max(0, (subhalo_offsets[subhalo_id, particle_type] > offsets[:, particle_type]).sum() - 1)  # first chunk
    start = subhalo_offsets[subhalo_id, particle_type] - offsets[chunk, particle_type]
    subhalo_length = subhalo_offsets[subhalo_id + 1, particle_type] - subhalo_offsets[subhalo_id, particle_type]
    if subhalo_length == 0:
        return None, None
    chunk_length = offsets[min(599, chunk + 1), particle_type] - offsets[min(598, chunk), particle_type]
    length = min(subhalo_length, chunk_length - start)
    remaining = subhalo_length - length
    i = 0
    while length > 0:
        snapshot_datapath = os.path.join(snapshot_dir, f"snap_{snapshot_id:03d}.{chunk + i}.hdf5")
        with h5py.File(snapshot_datapath, "r") as f:
            coords.append(f[f"PartType{particle_type:d}"]["Coordinates"][start:start + length, :] / 1e3)
            if particle_type != 1:
                mass.append(f[f"PartType{particle_type:d}"]["Masses"][start:start + length])
        i += 1
        start = 0
        chunk_length = offsets[min(599, chunk + 1 + i), particle_type] - offsets[min(598, chunk + i), particle_type]
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


parser = ArgumentParser()
# parser.add_argument("--groupcat", default="/home/aadam/projects/rrg-lplevass/aadam/illustrisTNG300-1_snapshot99_groupcat")
parser.add_argument("--output_dir", required=True, type=str, help="Directory where to save raster images (fits file)")
parser.add_argument("--subhalo_id", required=True, type=str,
                    help="npy file that contains array of int32 index of subhalos to rasterize")
parser.add_argument("--projection", required=True, type=str, help="2 characters, a combination of x, y and z (e.g. 'xy')")
parser.add_argument("--base_filenames", default="kappa")
parser.add_argument("--pixels", default=512, type=int, help="Number of pixels in the raster image")
parser.add_argument("--n_neighbors", default=10, type=int, help="Number of neighbors used to compute kernel length")
parser.add_argument("--offsets", default="/home/aadam/scratch/data/TNG300-1/offsets/offsets_099.hdf5")
parser.add_argument("--groupcat_dir", default="/home/aadam/projects/rrg-lplevass/aadam/illustrisTNG300-1_snapshot99_groupcat/")
parser.add_argument("--snapshot_dir", default="/home/aadam/scratch/data/TNG300-1/snapshot99/", help="Root directory of the snapshot")
parser.add_argument("--snapshot_id", default=99, type=int,
                    help="Should match id of snapshot given in snapshot argument")
parser.add_argument("--fov", default=1, type=float, help="Field of view of a scene in comoving Mpc")
parser.add_argument("--box_size", default=205, type=float, help="Box size of the simulation, in Mpc")
parser.add_argument("--z_source", default=1.5, type=float)
parser.add_argument("--z_lens", default=0.5, type=float)
parser.add_argument("--smoke_test", action="store_true")
args = parser.parse_args()

subhalo_ids = np.load(args.subhalo_id)
if args.smoke_test:
    print("smoke_test")
    subhalo_ids = np.array([10])  # should be an easy halo to try out
    print(subhalo_ids.size)
if "done.txt" in os.listdir(args.output_dir):   # for checkpointing
    done = np.loadtxt(os.path.join(args.output_dir, "done.txt"))
else:
    done = []
dims = projection(args.projection)

zd = args.z_lens
zs = args.z_source
Ds = cosmo.angular_diameter_distance(zs)
Dd = cosmo.angular_diameter_distance(zd)
Dds = cosmo.angular_diameter_distance_z1z2(zd, zs)
sigma_crit = (c ** 2 * Ds / (4 * np.pi * G * Dd * Dds) / (1e10 * M_sun)).to(u.Mpc ** (-2)).value

with h5py.File(args.offsets, "r") as f:
    offsets = f["FileOffsets"]["SnapByType"][:]  # global particle number at the beginning of a chunk
    subhalo_offsets = f["Subhalo"]["SnapByType"][:]
    subhalo_fileoffsets = f["FileOffsets"]["Subhalo"][:]


def distributed_strategy():
    for i in range(this_worker-1, subhalo_ids.size, N_WORKERS):
        subhalo_id = subhalo_ids[i]
        if subhalo_id in done:
            continue

        print(f"Started subhalo {subhalo_id} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")

        coords = []
        mass = []
        _len = []
        # load particles (0=gas, 1=DM, 4=stars, 5=black holes)
        for part_type in [0, 1, 4, 5]:
            coords_, mass_ = load_subhalo(subhalo_id, part_type, offsets, subhalo_offsets, args.snapshot_dir, args.snapshot_id)
            if coords_ is None:
                _len.append(0)
                continue
            _len.append(coords_.shape[0])
            coords.append(coords_)
            mass.append(mass_)
        coords = np.concatenate(coords)
        mass = np.concatenate(mass)

        # load subhalo position for fixed_boundary_coordinates
        chunk = max(0, (subhalo_id > subhalo_fileoffsets).sum() - 1)  # first chunk
        subhalo_index = subhalo_id - subhalo_fileoffsets[chunk]
        chunk_length = subhalo_fileoffsets[min(599, chunk + 1)] - subhalo_fileoffsets[min(598, chunk)]
        while subhalo_index >= chunk_length:
            chunk += 1
            subhalo_index -= chunk_length
            chunk_length = subhalo_fileoffsets[min(599, chunk + 1)] - subhalo_fileoffsets[min(598, chunk)]
        fof_datapath = os.path.join(args.groupcat_dir, f"fof_subhalo_tab_099.{chunk}.hdf5")
        with h5py.File(fof_datapath, "r") as f:
            centroid = f["Subhalo"]["SubhaloCM"][subhalo_index] / 1e3

        coords = fixed_boundary_coordinates(coords, centroid, args.box_size)
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
        margin = args.fov / args.pixels # allow for particle near the margin to be counted in
        select = (x < xmax + margin) & (x > xmin - margin) & (y < ymax + margin) & (y > ymin - margin)
        kappa = gaussian_kernel_rasterize(coords[select], mass[select], center, args.fov, dims=dims, pixels=args.pixels,
                                          n_neighbors=args.n_neighbors)
        kappa /= sigma_crit

        # create fits file and its header, than save result
        date = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        print(f"Finished subhalo {subhalo_id} at {date}")

        header = fits.Header()
        header["SUBID"] = subhalo_id
        header["MASS"] = (mass.sum(), "Total mass in the subhalo, in 10^{10} solar mass units")
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
        header["SIGCRIT"] = sigma_crit
        header["COSMO"]  = "Planck18"
        hdu = fits.PrimaryHDU(kappa, header=header)
        hdul = fits.HDUList([hdu])
        hdul.writeto(os.path.join(args.output_dir,
                                  args.base_filenames + f"_{subhalo_id:06d}_{args.projection}.fits"))
        print("Finished")


if __name__ == '__main__':
    distributed_strategy()
