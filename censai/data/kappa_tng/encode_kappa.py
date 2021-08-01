import tensorflow as tf
from censai.utils import _bytes_feature, _float_feature, _int64_feature


def encode_examples(
        kappa: tf.Tensor,
        einstein_radius_init: list,
        einstein_radius: list,
        rescalings: list,
        z_source: float,
        z_lens: float,
        kappa_fov: float,
        sigma_crit: float,
        kappa_ids: list
):
    batch_size = kappa.shape[0]
    kappa_pixels = kappa.shape[1]
    records = []
    for j in range(batch_size):
        features = {
            "kappa": _bytes_feature(kappa[j].numpy().tobytes()),
            "Einstein radius before rescaling": _float_feature(einstein_radius_init[j]),
            "Einstein radius": _float_feature(einstein_radius[j]),
            "rescaling factor": _float_feature(rescalings[j]),
            "z source": _float_feature(z_source),
            "z lens": _float_feature(z_lens),
            "kappa fov": _float_feature(kappa_fov),    # arc seconds
            "sigma crit": _float_feature(sigma_crit),  # 10^10 M_sun / Mpc^2
            "kappa pixels": _int64_feature(kappa_pixels),
            "kappa id": _int64_feature(kappa_ids[j])
        }
        serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
        record = serialized_output.SerializeToString()
        records.append(record)
    return records
