import tensorflow as tf
import os
from datetime import datetime
import galsim
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from numpy.lib.recfunctions import append_fields
import numpy as np

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


def distributed_strategy(args):
    print(f"Started worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")
    options = tf.io.TFRecordOptions(compression_type=args.compression_type)

    catalog = galsim.COSMOSCatalog(sample=args.sample, dir=args.cosmos_dir, exclusion_level=args.exclusion_level, min_flux=args.min_flux)
    n_galaxies = catalog.getNObjects()
    cat_param = catalog.param_cat[catalog.orig_index]
    sparams = cat_param['sersicfit']
    cat_param = append_fields(cat_param, 'sersic_q', sparams[:, 3])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:, 2])

    with tf.io.TFRecordWriter(os.path.join(args.output_dir, f"data_{THIS_WORKER}.tfrecords"), options) as writer:
        for index in range((THIS_WORKER - 1), n_galaxies, N_WORKERS):
            gal = catalog.makeGalaxy(index, noise_pad_size=args.pixels * args.pixel_scale)
            psf = gal.original_psf

            # Apply random rotation if requested
            if hasattr(args, "rotation") and args.rotation:
                rotation_angle = galsim.Angle(-np.random.rand() * 2 * np.pi,
                                              galsim.radians)
                gal = gal.rotate(rotation_angle)
                psf = psf.rotate(rotation_angle)

            # We save the corresponding attributes for this galaxy
            if hasattr(args, 'attributes'):
                params = cat_param[index]
                attributes = {k: params[k] for k in args.attributes}
            else:
                attributes = None

            # Apply the PSF
            gal = galsim.Convolve(gal, psf)

            # Compute sqrt of absolute noise power spectrum, at the resolution and stamp size of target image
            ps = gal.noise._get_update_rootps((args.pixels, args.pixels), wcs=galsim.PixelScale(args.pixel_scale))

            # We draw the pixel image of the convolved image
            im = gal.drawImage(nx=args.pixels, ny=args.pixels, scale=args.pixel_scale,
                               method='no_pixel', use_true_center=False).array.astype('float32')

            # preprocess image
            # For correlated noise, we estimate that the sqrt of the Energy Spectral Density of the noise at (f_x=f_y=0)
            # is a good estimate of the STD
            # Background is interpreted as the sqrt(Var(im)).
            im = tf.nn.relu(im - ps[0, 0]).numpy()       # subtract backgroung, fold negative pixels to 0
            im /= im.sum()                               # normalize peak to 1
            signal_pixels = np.sum(im > args.signal_threshold)     # how many pixels have a value above a certain threshold
            if signal_pixels < args.signal_pixels:  # argument used to select only examples that are more distinct galaxy features (it does however bias the dataset in redshift space)
                continue

            # Draw a kimage of the galaxy, just to figure out what maxk is, there might
            # be more efficient ways to do this though...
            bounds = galsim.BoundsI(0, args.pixels // 2, -args.pixels // 2, args.pixels // 2 - 1)
            imG = gal.drawKImage(bounds=bounds,
                                 scale=2. * np.pi / (args.pixels * args.pixels),
                                 recenter=False)
            mask = ~(np.fft.fftshift(imG.array, axes=0) == 0)

            # Draw the Fourier domain image of the galaxy, using x1 zero padding,
            # and x2 subsampling
            interp_factor = 2
            padding_factor = 1
            Nk = args.pixels * interp_factor * padding_factor
            bounds = galsim.BoundsI(0, Nk // 2, -Nk // 2, Nk // 2 - 1)
            imCp = psf.drawKImage(bounds=bounds,
                                  scale=2. * np.pi / (Nk * args.pixel_scale / interp_factor),
                                  recenter=False)

            # Transform the psf array into proper format, remove the phase
            im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')

            # The following comes from correlatednoise.py
            rt2 = np.sqrt(2.)
            shape = (args.pixels, args.pixels)
            ps[0, 0] = rt2 * ps[0, 0]
            # Then make the changes necessary for even sized arrays
            if shape[1] % 2 == 0:  # x dimension even
                ps[0, shape[1] // 2] = rt2 * ps[0, shape[1] // 2]
            if shape[0] % 2 == 0:  # y dimension even
                ps[shape[0] // 2, 0] = rt2 * ps[shape[0] // 2, 0]
                # Both dimensions even
                if shape[1] % 2 == 0:
                    ps[shape[0] // 2, shape[1] // 2] = rt2 * ps[shape[0] // 2, shape[1] // 2]

            # Apply mask to power spectrum so that it is very large outside maxk
            ps = np.where(mask, np.log(ps ** 2), 10).astype('float32')
            features = {
                "image": _bytes_feature(im.tobytes()),
                "height": _int64_feature(im.shape[0]),
                "width": _int64_feature(im.shape[1]),
                "psf": _bytes_feature(im_psf.tobytes()),
                "ps": _bytes_feature(ps.tobytes()),  # power spectrum of the noise
            }

            # Adding the parameters provided
            if attributes is not None:
                for k in attributes:
                    features['attrs_' + k] = _float_feature(attributes[k])

            record = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
            writer.write(record)
    print(f"Finished worker {THIS_WORKER} at {datetime.now().strftime('%y-%m-%d_%H-%M-%S')}")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--pixel_scale",            default=0.03,       type=float, help="Native pixel resolution of the image")
    parser.add_argument("--pixels",                 default=128,        type=int,   help="Number of pixels on a side of the drawn postage stamp")
    parser.add_argument("--sample",                 default="25.2",                 help="Either 25.2 or 23.5")
    parser.add_argument("--exclusion_level",        default="marginal",             help="Galsim exclusion level of bad postage stamps")
    parser.add_argument("--min_flux",               default=0.,         type=float, help="Minimum flux for the original postage stamps")
    parser.add_argument("--signal_pixels",          default=0,          type=int,   help="Minimal number of pixel with value abover user defined signal threshold -- after peak is normalized to 1")
    parser.add_argument("--signal_threshold",       default=0,          type=float, help="Value between 0 and 1, defines the pixel value below which there is no more signal")
    parser.add_argument("--cosmos_dir",             default=None,                   help="Directory to cosmos data")
    parser.add_argument("--store_attributes",       action="store_true",            help="Wether to store ['mag_auto', 'flux_radius', 'sersic_n', 'sersic_q', 'z_phot] or not")
    parser.add_argument("--rotation",               action="store_true",            help="Rotate randomly the postage stamp (and psf)")
    parser.add_argument("--output_dir",             required=True,                  help="Path to the directory where to store tf records")
    parser.add_argument("--compression_type",       default=None,                   help="Default is no compression. Use 'GZIP' to compress")

    args = parser.parse_args()
    if args.store_attributes:
        vars(args)["attributes"] = ['mag_auto', 'flux_radius', 'sersic_n', 'sersic_q', 'zphot']
    if not os.path.isdir(args.output_dir) and THIS_WORKER <= 1:
        os.mkdir(args.output_dir)

    distributed_strategy(args)