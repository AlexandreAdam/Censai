import tensorflow as tf
import numpy as np
import io
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def plot_to_image(figure):
      """
      Converts the matplotlib plot specified by 'figure' to a PNG image and
      returns it. The supplied figure is closed and inaccessible after this call.
      """
      buf = io.BytesIO()
      plt.savefig(buf, format='png')
      plt.close(figure)
      buf.seek(0)
      # Convert PNG buffer to TF image
      image = tf.image.decode_png(buf.getvalue(), channels=4)
      # Add the batch dimension
      image = tf.expand_dims(image, 0)
      return image


def deflection_angles_residual_plot(y_true, y_pred):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(2):
        im = axs[i, 0].imshow(y_true.numpy()[..., i], cmap="jet", origin="lower")
        divider = make_axes_locatable(axs[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 0].axis("off")

        im = axs[i, 1].imshow(y_pred.numpy()[..., i], cmap="jet", origin="lower")
        divider = make_axes_locatable(axs[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 1].axis("off")

        residual = np.abs(y_true.numpy()[..., i] - y_pred.numpy()[..., i])
        im = axs[i, 2].imshow(residual, cmap="jet", origin="lower")
        divider = make_axes_locatable(axs[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 2].axis("off")

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title("Prediction")
    axs[0, 2].set_title("Residual")
    plt.subplots_adjust(wspace=.2, hspace=.0)
    plt.figtext(0.1, 0.7, r"$\alpha_x$", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.3, r"$\alpha_y$", va="center", ha="center", size=15, rotation=90)
    return fig