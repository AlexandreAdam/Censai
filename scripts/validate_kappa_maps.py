from astropy.io import fits
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LogStretch
import os, glob
from argparse import ArgumentParser
import math


def visualize(args):
    for file in tqdm(glob.glob(os.path.join(args.kappa_dir, "*_xy.fits"))):
        try:
            a = fits.open(file)
            id = a[0].header["SUBID"]
        except OSError:
            print(file)
            continue
        mass = a[0].header["CUTMASS"]
        if a[0].data.max() > args.kappa_cutoff and np.random.uniform() < args.random_cutoff:
            continue
        plt.figure(figsize=(10, 10))
        plt.imshow(a[0].data, cmap=args.cmap, norm=ImageNormalize(stretch=LogStretch()))
        plt.colorbar()
        plt.axis("off")
        plt.title(f"mass={mass:.2f}, id={id}")
        plt.contour(a[0].data, levels=[0.1, 0.5, 1])
        plt.show()


def main(args):
    output_file_path = os.path.join(args.kappa_dir, "good_kappa.txt")
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
    for file in tqdm(glob.glob(os.path.join(args.kappa_dir, "*_xy.fits"))):
        try:
            a = fits.open(file)
            id = a["PRIMARY"].header["SUBID"]
            kappa_max = a["PRIMARY"].data.max()
            if kappa_max > 1:
                with open(output_file_path, "a") as f:
                    f.write(f"{id:08d}\n")

        except OSError:
            print(file)
            continue

    good_kappa_index = np.loadtxt(output_file_path)
    train_size = math.floor(args.train_split * good_kappa_index.size)
    train_index = np.random.choice(good_kappa_index, size=train_size, replace=False)
    test_index = np.array(list(set(good_kappa_index).difference(train_index)))
    np.savetxt(fname=os.path.join(args.kappa_dir, "train_kappa.txt"), X=train_index, fmt="%d")
    np.savetxt(fname=os.path.join(args.kappa_dir, "test_kappa.txt"), X=test_index, fmt="%d")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kappa_dir", required=True, help="Path to directory of kappa maps")

    # options for visualize
    parser.add_argument("-r", "--random_cutoff", default=0.80, type=float,
                        help="Uniform random variable > r will trigger a valid kappa map to show")
    parser.add_argument("-k", "--kappa_cutoff", default=2, type=float,
                        help="If kappa maps reach this value at least at one pixel, it is considered valid")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--cmap", default="magma")

    # options for main
    parser.add_argument("--good_kappa_cutoff", default=1, type=float,
                             help="Threshold for the maximum value of a pixel in kappa map, below which "
                                  "we discard the map.")

    parser.add_argument("--train_split", default=0.9, type=float, help="At the end of the script, split validated kappa maps into a training set and a test set")

    args = parser.parse_args()

    if args.visualize:
        visualize(args)
    else:
        main(args)
