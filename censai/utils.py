import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pickle
import collections
try:
    from contextlib import nullcontext  # python > 3.7 needed for this
except ImportError:
    # Backward compatibility with python <= 3.6
    class nullcontext:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


class nullwriter:
    @staticmethod
    def flush():
        pass

    @staticmethod
    def as_default():
        return nullcontext()


def convert_to_8_bit(image):
    return (255.0 * image).astype(np.uint8)


def convert_to_float(image):
    "normalize image from uint8 to float32"
    return tf.cast(image, tf.float32)/255.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def save_output(output, dirname, epoch, batch, index_mod, epoch_mod, timestep_mod, format="png", filename_base="output"):
    if epoch % epoch_mod != 0:
        return
    out = output
    if tf.is_tensor(out):
        out = output.numpy()
    if len(out.shape) == 5:
        # parallelize search for the image
        image_index = np.arange(out.shape[0])
        true_image_index = image_index + batch * out.shape[0]
        image_index = image_index[(true_image_index) % index_mod == 0]
        timestep = np.arange(out.shape[-1])
        timestep = timestep[(timestep + 1) % timestep_mod == 0]
        timestep = np.tile(timestep, reps=[image_index.size, 1])  # fancy broadcasting of the indices
        image_index = np.tile(image_index, reps=[timestep.shape[1], 1])
        for i, I in enumerate(out[image_index.T, ..., timestep]): # note that array is reshaped to [batch, steps, pix, pix, channel]
            for j, image in enumerate(I[..., 0]):
                if format == "png":
                    image = convert_to_8_bit(image)
                    image = Image.fromarray(image, mode="L")
                    image.save(os.path.join(dirname, f"{filename_base}_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.png"))
                elif format == "txt":
                    np.savetxt(os.path.join(dirname, f"{filename_base}_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.txt"), image)
    elif len(out.shape) == 4:
        for timestep in range(out.shape[-1]):
            if timestep % timestep_mod == 0:
                continue
            image = convert_to_8_bit(out[:, :, 0, timestep])
            image = Image.fromarray(image, mode="L")
            image.save(os.path.join(dirname, f"output_{epoch:03}_000_{timestep:02}.png"))


def save_gradient_and_weights(grad, trainable_weight, dirname, epoch, batch):
    file = os.path.join(dirname, "grad_and_weights.pickle")
    if os.path.exists(file):
        with open(file, "rb") as f:
            d = pickle.load(f)
    else:
        d = {}
    for i, _ in enumerate(grad):
        layer = trainable_weight[i].name
        update(d, {layer : {epoch : {batch : {
            "grad_mean": grad[i].numpy().mean(),
            "grad_var" : grad[i].numpy().std(),
            "weight_mean": trainable_weight[i].numpy().mean(),
            "weight_var": trainable_weight[i].numpy().std(),
            "weight_max": trainable_weight[i].numpy().max(),
            "weight_min": trainable_weight[i].numpy().min()
        }}}})
    with open(file, "wb") as f:
        pickle.dump(d, f)


def save_loglikelihood_grad(grad, dirname, epoch, batch, index_mod, epoch_mod, timestep_mod, step_mod):
    if epoch % epoch_mod != 0:
        return
    out = grad.numpy()
    image_index = np.arange(out.shape[0])
    true_image_index = image_index + batch * out.shape[0]
    image_index = image_index[(true_image_index) % index_mod == 0]
    timestep = np.arange(out.shape[-1])
    timestep = timestep[(timestep + 1) % timestep_mod == 0]
    timestep = np.tile(timestep, reps=[image_index.size, 1])
    image_index = np.tile(image_index, reps=[timestep.shape[1], 1])
    for i, G in enumerate(out[image_index.T, ..., timestep]):
        for j, g in enumerate(G):
            np.savetxt(os.path.join(dirname, f"grad_{epoch:04}_{true_image_index[i]:04}_{timestep[i, j]:02}.txt"), g)
