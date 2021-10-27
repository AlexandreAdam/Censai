import tensorflow as tf
import numpy as np
from censai.utils import _bytes_feature, _float_feature, _int64_feature


def encode_examples(
        kappa: tf.Tensor,
        galaxies: tf.Tensor,
        lensed_images: tf.Tensor,
        z_source: float,
        z_lens: float,
        image_fov: float,
        kappa_fov: float,
        source_fov: float,
        noise_rms: np.array,
        psf: tf.Tensor,
):
    batch_size = galaxies.shape[0]
    source_pixels = galaxies.shape[1]
    kappa_pixels = kappa.shape[1]
    pixels = lensed_images.shape[1]
    psf_pixels=  psf.shape[1]
    records = []
    for j in range(batch_size):
        features = {
            "kappa": _bytes_feature(kappa[j].numpy().tobytes()),
            "source": _bytes_feature(galaxies[j].numpy().tobytes()),
            "lens": _bytes_feature(lensed_images[j].numpy().tobytes()),
            "z source": _float_feature(z_source),
            "z lens": _float_feature(z_lens),
            "image fov": _float_feature(image_fov),    # arc seconds
            "kappa fov": _float_feature(kappa_fov),    # arc seconds
            "source fov": _float_feature(source_fov),  # arc seconds
            "src pixels": _int64_feature(source_pixels),
            "kappa pixels": _int64_feature(kappa_pixels),
            "pixels": _int64_feature(pixels),
            "noise rms": _float_feature(noise_rms[j]),
            "psf": _bytes_feature(psf[j].numpy().tobytes()),
            "psf pixels": _int64_feature(psf_pixels)
        }
        serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
        record = serialized_output.SerializeToString()
        records.append(record)
    return records