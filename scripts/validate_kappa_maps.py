from astropy.io import fits
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LogStretch
import os

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--cmap", default="magma")
parser.add_argument("-r", "--random_cutoff", default=0.98, type=float, help="Uniform random variable > r will trigger a valid kappa map to show")
parser.add_argument("-k", "--kappa_cutoff", default=2, type=float, help="If kappa maps reach this value at least at one pixel, it is considered valid")
parser.add_argument("--kappa_dir", required=True, help="Path to directory of kappa maps")
args = parser.parse_args()

bad_kappa_ids = np.loadtxt("../data/bad_kappa_ids.txt") # already identified bad kappa maps

id_prev = None
for file in tqdm(os.listdir(args.kappa_dir)):
    a = fits.open(os.path.join(args.kappa_dir, file))
    id = a[0].header["SUBID"]
    if id in bad_kappa_ids:
        continue
    if a[0].data.max() > args.kappa_cutoff and np.random.uniform() < args.random_cutoff:
        continue
    plt.imshow(a[0].data, cmap=args.cmap, norm=ImageNormalize(stretch=LogStretch()))
    plt.colorbar()
    plt.axis("off")
    plt.title(f"id_prev={id_prev}, id={id}")
    plt.contour(a[0].data, levels=[0.1, 0.5, 1])
    plt.show()
    id_prev = id