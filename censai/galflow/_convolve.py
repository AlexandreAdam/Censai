"""
Author: Fançois Lanusse
Package: Galflow: https://github.com/DifferentiableUniverseInitiative/GalFlow
"""
import numpy as np
import tensorflow as tf


def k_wrapping(kimage, wrap_factor=2):
    """
    Wraps kspace image of a real image to decrease its resolution by specified
    factor
    """
    batch_size, Nkx, Nky = kimage.get_shape().as_list()

    # First wrap around the non hermitian dimension
    rkimage = kimage + tf.roll(kimage, shift=Nkx//wrap_factor, axis=1)

    # Now take care of the hermitian part
    revrkimage = tf.reverse(tf.math.conj(tf.reverse(rkimage, axis=[2])), axis=[1])

    # These masks take care of the special case of the 0th frequency
    mask = np.ones((1, Nkx, Nky))
    mask[:,0,:]=0
    mask[:,Nkx//wrap_factor-1,:]=0
    rkimage2 = rkimage + revrkimage*mask
    mask = np.zeros((1, Nkx, Nky))
    mask[:,Nkx//wrap_factor-1,:]=1
    rkimage2 = rkimage2 + tf.roll(revrkimage,shift=-1, axis=1) *mask

    # Now that we have wrapped the image, we can truncate it to the desired size
    kimage = rkimage2[:, :Nkx//wrap_factor, :(Nky-1)//wrap_factor+1]

    return kimage


def kconvolve(kimages, kpsf, zero_padding_factor=2, interp_factor=2):
    """
    Convolution of provided k-space images and psf tensor.
    Careful! This function doesn't remove zero padding.
    Careful! When using a kimage and kpsf from GalSim,
           one needs to apply an fftshift at the output.
    This function assumes that the k-space tensors are already prodided with the
    stepk and maxk corresponding to the specified interpolation and zero padding
    factors.
    """
    batch_size, Nkx, Nky = kimages.get_shape().as_list()
    Nx = Nkx // zero_padding_factor // interp_factor

    # Perform k-space convolution
    imk = kimages * kpsf

    # Apply frequency wrapping to reach target image resolution
    if interp_factor >1:
        imk = k_wrapping(imk, interp_factor)

    # Perform inverse Fourier Transform
    conv_images = tf.signal.irfft2d(imk)

    # Rebuilds channels, but doesnt remove zero padding
    conv_images = tf.expand_dims(conv_images, axis=-1)

    return conv_images


def convolve(
        images,
        kpsf,
        x_interpolant=tf.image.ResizeMethod.BICUBIC,
        zero_padding_factor=2,
        interp_factor=2):
    """
    Convolution of input images with provided k-space psf tensor.
    This function assumes that the k-space PSF is already prodided with the
    stepk and maxk corresponding to the specified interpolation and zero padding
    factors.
    """
    batch_size, Nx, Ny, Nc = images.get_shape().as_list()
    # For now, we only support square, even images, in only one band
    assert Nx == Ny
    assert Nx % 2 == 0
    assert Nc == 1

    # First, we interpolate the image on a finer grid
    if interp_factor > 1:
        im = tf.image.resize(images, [Nx*interp_factor, Ny*interp_factor], method=x_interpolant)
    else:
        im = images
    # Since we lower the resolution of the image, we also scale the flux
    # accordingly
    images = im / interp_factor**2

    # Second, we pad as necessary
    im = tf.image.resize_with_crop_or_pad(images,
                                          zero_padding_factor*Nx*interp_factor,
                                          zero_padding_factor*Ny*interp_factor)

    # Compute DFT
    imk = tf.signal.rfft2d(im[..., 0])

    # Performing k space convolution
    imconv = kconvolve(imk, kpsf,
                       zero_padding_factor=zero_padding_factor,
                       interp_factor=interp_factor)

    # Removing zero padding
    return tf.image.resize_with_crop_or_pad(imconv, Nx, Nx)
