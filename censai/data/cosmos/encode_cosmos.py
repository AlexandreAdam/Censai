import numpy as np
import tensorflow as tf
import galsim
from censai.utils import _bytes_feature, _float_feature, _int64_feature
from numpy.lib.recfunctions import append_fields


def draw_and_encode_stamp(gal, psf, stamp_size, pixel_scale, attributes=None):
    """
    Draws the galaxy, psf and noise power spectrum on a postage stamp and
    encodes it to be exported in a TFRecord.

    Taken from galaxy2galaxy by François Lanusse https://github.com/ml4astro/galaxy2galaxy
    Modified by Alexandre Adam May 29, 2021
    """

    # Apply the PSF
    gal = galsim.Convolve(gal, psf)

    # Draw a kimage of the galaxy, just to figure out what maxk is, there might
    # be more efficient ways to do this though...
    bounds = galsim.BoundsI(0, stamp_size//2, -stamp_size//2, stamp_size//2-1)
    imG = gal.drawKImage(bounds=bounds,
                         scale=2.*np.pi/(stamp_size * pixel_scale),
                         recenter=False)
    mask = ~(np.fft.fftshift(imG.array, axes=0) == 0)

    # We draw the pixel image of the convolved image
    im = gal.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale,
                       method='no_pixel', use_true_center=False).array.astype('float32')

    # Draw the Fourier domain image of the galaxy, using x1 zero padding,
    # and x2 subsampling
    interp_factor = 2
    padding_factor = 1
    Nk = stamp_size*interp_factor*padding_factor
    bounds = galsim.BoundsI(0, Nk//2, -Nk//2, Nk//2-1)
    imCp = psf.drawKImage(bounds=bounds,
                          scale=2.*np.pi/(Nk * pixel_scale / interp_factor),
                          recenter=False)

    # Transform the psf array into proper format, remove the phase
    im_psf = np.abs(np.fft.fftshift(imCp.array, axes=0)).astype('float32')

    # Compute noise power spectrum, at the resolution and stamp size of target
    # image
    ps = gal.noise._get_update_rootps((stamp_size, stamp_size), wcs=galsim.PixelScale(pixel_scale))

    # The following comes from correlatednoise.py
    rt2 = np.sqrt(2.)
    shape = (stamp_size, stamp_size)
    ps[0, 0] = rt2 * ps[0, 0]
    # Then make the changes necessary for even sized arrays
    if shape[1] % 2 == 0:  # x dimension even
        ps[0, shape[1] // 2] = rt2 * ps[0, shape[1] // 2]
    if shape[0] % 2 == 0:  # y dimension even
        ps[shape[0] // 2, 0] = rt2 * ps[shape[0] // 2, 0]
        # Both dimensions even
        if shape[1] % 2 == 0:
            ps[shape[0] // 2, shape[1] // 2] = rt2 * \
                ps[shape[0] // 2, shape[1] // 2]

    # Apply mask to power spectrum so that it is very large outside maxk
    ps = np.where(mask, np.log(ps**2), 10).astype('float32')
    features = {
        "image": _bytes_feature(im.tobytes()),
        "height": _int64_feature(im.shape[0]),
        "width": _int64_feature(im.shape[1]),
        "psf": _bytes_feature(im_psf.tobytes()),
        "ps": _bytes_feature(ps.tobytes()),  # power spectrum of the noise
        # "ps_height": _int64_feature(ps.shape[0]),
        # "ps_width": _int64_feature(ps.shape[1])
    }

    # Adding the parameters provided
    if attributes is not None:
        for k in attributes:
            features['attrs_'+k] = _float_feature(attributes[k])

    serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
    return serialized_output.SerializeToString()


def encode_examples(hparams, task_id=0, sample="25.2", cosmos_dir=None, exclusion_level="marginal", min_flux=0.):
    """
    Generates and yields postage stamps obtained with GalSim.
    """
    catalog = galsim.COSMOSCatalog(sample=sample, dir=cosmos_dir, exclusion_level=exclusion_level, min_flux=min_flux)

    index = range(task_id * hparams.example_per_shard,
                  min((task_id+1) * hparams.example_per_shard, catalog.getNObjects()))

    # Extracts additional information about the galaxies
    cat_param = catalog.param_cat[catalog.orig_index]
    # bparams = cat_param['bulgefit']
    sparams = cat_param['sersicfit']

    cat_param = append_fields(cat_param, 'sersic_q', sparams[:, 3])
    cat_param = append_fields(cat_param, 'sersic_n', sparams[:, 2])

    for ind in index:
        # Draw a galaxy using GalSim, any kind of operation can be done here
        gal = catalog.makeGalaxy(ind, noise_pad_size=hparams.img_len * hparams.pixel_scale)
        psf = gal.original_psf

        # Apply random rotation if requested
        if hasattr(hparams, "rotation") and hparams.rotation:
            rotation_angle = galsim.Angle(-np.random.rand() * 2 * np.pi,
                                          galsim.radians)
            gal = gal.rotate(rotation_angle)
            psf = psf.rotate(rotation_angle)

        # We save the corresponding attributes for this galaxy
        if hasattr(hparams, 'attributes'):
            params = cat_param[ind]
            attributes = {k: params[k] for k in hparams.attributes}
        else:
            attributes = None

        # Utility function encodes the postage stamp for serialized features
        yield draw_and_encode_stamp(gal, psf,
                                    stamp_size=hparams.img_len,
                                    pixel_scale=hparams.pixel_scale,
                                    attributes=attributes)