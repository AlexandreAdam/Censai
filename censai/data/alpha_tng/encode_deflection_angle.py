import tensorflow as tf
from censai.utils import _bytes_feature, _float_feature, _int64_feature


def encode_examples(
        kappa: tf.Tensor,
        alpha: tf.Tensor,
        rescalings: list,
        kappa_ids: list,
        einstein_radius: list,
        image_fov: float,
        kappa_fov: float
):
    batch_size = kappa.shape[0]
    pixels = kappa.shape[1]
    records = []
    for j in range(batch_size):
        features = {
            "kappa": _bytes_feature(kappa[j].numpy().tobytes()),
            "pixels": _int64_feature(pixels),
            "alpha": _bytes_feature(alpha[j].numpy().tobytes()),
            "rescale": _float_feature(rescalings[j]),
            "kappa_id": _int64_feature(kappa_ids[j]),
            "Einstein radius": _float_feature(einstein_radius[j]),
            "image_fov": _float_feature(image_fov),
            "kappa_fov": _float_feature(kappa_fov)
        }

        serialized_output = tf.train.Example(features=tf.train.Features(feature=features))
        record = serialized_output.SerializeToString()
        records.append(record)
    return records
