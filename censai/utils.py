import tensorflow as tf
import numpy as np
import io
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
import re

try:
    from contextlib import nullcontext  # python > 3.7 needed for this
except ImportError:
    # Backward compatibility with python <= 3.6
    class nullcontext:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class nullwriter:
    @staticmethod
    def flush():
        pass

    @staticmethod
    def as_default():
        return nullcontext()


class nulltape(nullcontext):
    @staticmethod
    def stop_recording():
        return nullcontext()

    @staticmethod
    def flush():
        pass


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
        im = axs[i, 0].imshow(y_true[..., i], cmap="jet", origin="lower")
        divider = make_axes_locatable(axs[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 0].axis("off")

        im = axs[i, 1].imshow(y_pred[..., i], cmap="jet", origin="lower")
        divider = make_axes_locatable(axs[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 1].axis("off")

        residual = np.abs(y_true[..., i] - y_pred[..., i])
        im = axs[i, 2].imshow(residual, cmap="seismic", norm=CenteredNorm(), origin="lower")
        divider = make_axes_locatable(axs[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 2].axis("off")

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title("Prediction")
    axs[0, 2].set_title("Residual")
    plt.subplots_adjust(wspace=.2, hspace=.2)
    plt.figtext(0.1, 0.7, r"$\alpha_x$", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.3, r"$\alpha_y$", va="center", ha="center", size=15, rotation=90)
    return fig


def lens_residual_plot(lens_true, lens_pred, title=""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    ax = axs[0]
    im = ax.imshow(lens_true[..., 0], cmap="hot",  origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[1]
    im = ax.imshow(lens_pred[..., 0], cmap="hot", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2]
    im = ax.imshow((lens_true - lens_pred)[..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    fig.suptitle(f"{title}", size=20)
    axs[0].set_title("Ground Truth", size=15)
    axs[1].set_title("Predictions", size=15)
    axs[2].set_title("Residuals", size=15)
    plt.subplots_adjust(wspace=.2, hspace=.2)
    return fig


def raytracer_residual_plot(y_true, y_pred, lens_true, lens_pred):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    for i in range(2):
        im = axs[i, 0].imshow(y_true[..., i], cmap="seismic", norm=CenteredNorm(), origin="lower")
        divider = make_axes_locatable(axs[i, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 0].axis("off")

        im = axs[i, 1].imshow(y_pred[..., i], cmap="seismic", norm=CenteredNorm(), origin="lower")
        divider = make_axes_locatable(axs[i, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 1].axis("off")

        residual = np.abs(y_true[..., i] - y_pred[..., i])
        im = axs[i, 2].imshow(residual, cmap="seismic", norm=CenteredNorm(), origin="lower")
        divider = make_axes_locatable(axs[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        axs[i, 2].axis("off")

    ax = axs[2, 0]
    im = ax.imshow(lens_true[..., 0], cmap="hot", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2, 1]
    im = ax.imshow(lens_pred[..., 0], cmap="hot", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2, 2]
    im = ax.imshow((lens_true - lens_pred)[..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].set_title("Prediction")
    axs[0, 2].set_title("Residual")
    plt.subplots_adjust(wspace=.2, hspace=.2)
    plt.figtext(0.1, 0.75, r"$\alpha_x$", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.5,  r"$\alpha_y$", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.25, r"Lens", va="center", ha="center", size=15, rotation=90)
    return fig


def rim_residual_plot(lens_true, source_true, kappa_true, lens_pred, source_pred, kappa_pred, chi_squared):
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    ax = axs[0, 0]
    im = ax.imshow(lens_true[..., 0], cmap="hot", origin="lower", vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[1, 0]
    im = ax.imshow(source_true[..., 0], cmap="bone", origin="lower", vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2, 0]
    im = ax.imshow(kappa_true[..., 0], cmap="hot", norm=LogNorm(vmin=1e-1, vmax=100), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[0, 1]
    im = ax.imshow(lens_pred[..., 0], cmap="hot", origin="lower", vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[1, 1]
    im = ax.imshow(source_pred[..., 0], cmap="bone", origin="lower", vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2, 1]
    im = ax.imshow(kappa_pred[..., 0], cmap="hot", norm=LogNorm(vmin=1e-1, vmax=100), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[0, 2]
    im = ax.imshow((lens_true - lens_pred)[..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[1, 2]
    im = ax.imshow((source_true - source_pred)[..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2, 2]
    im = ax.imshow((kappa_true - kappa_pred)[..., 0], cmap="seismic", norm=SymLogNorm(linthresh=1e-1, base=10, vmax=100, vmin=-100), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    axs[0, 0].set_title("Ground Truth", size=15)
    axs[0, 1].set_title("Predictions", size=15)
    axs[0, 2].set_title("Residuals", size=15)
    fig.suptitle(fr"$\chi^2$ = {chi_squared: .3e}", size=20)
    plt.subplots_adjust(wspace=.4, hspace=.2)
    plt.figtext(0.1, 0.75, r"Lens", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.5, r"Source", va="center", ha="center", size=15, rotation=90)
    plt.figtext(0.1, 0.22, r"$\kappa$", va="center", ha="center", size=15, rotation=90)

    return fig


def vae_residual_plot(y_true, y_pred):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    ax = axs[0]
    im = ax.imshow(y_true[..., 0], cmap="hot", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[1]
    im = ax.imshow(y_pred[..., 0], cmap="hot", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    ax = axs[2]
    im = ax.imshow(y_true[..., 0] - y_pred[..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.axis("off")

    axs[0].set_title("Ground Truth", size=15)
    axs[1].set_title("Prediction", size=15)
    axs[2].set_title("Residual", size=15)
    return fig


def reconstruction_plot(y_true, y_pred):
    batch_size = y_true.shape[0]
    len_y = batch_size // 3
    fig, axs = plt.subplots(len_y, 9, figsize=(27, 3 * len_y))

    for i in range(len_y):
        for j in range(3):
            k = (i * 3 + j) % batch_size
            ax = axs[i, j]
            ax.imshow(y_true[k, ..., 0], cmap="hot", origin="lower")
            ax.axis("off")

            ax = axs[i, j + 3]
            ax.imshow(y_pred[k, ..., 0], cmap="hot", origin="lower")
            ax.axis("off")

            ax = axs[i, j + 6]
            ax.imshow(y_true[k, ..., 0] - y_pred[k, ..., 0], cmap="seismic", norm=CenteredNorm(), origin="lower")
            ax.axis("off")

    axs[0, 1].set_title("Ground Truth", size=20)
    axs[0, 4].set_title("Prediction", size=20)
    axs[0, 7].set_title("Residual", size=20)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def sampling_plot(y):
    batch_size = y.shape[0]
    len_y = batch_size // 9
    fig, axs = plt.subplots(len_y, 9, figsize=(3 * len_y, 27))
    for i in range(len_y):
        for j in range(9):
            k = 9 * i + j
            axs[i, j].imshow(y[k, ..., 0], cmap="hot", origin="lower")
            axs[i, j].axis("off")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

# def rim_analytic_residual_plot(phys, rim, batch_size, total_preds):
#     x_preds = []
#     x_true = []
#     chi_sq = []
#     total_preds = total_preds//batch_size * batch_size
#     for i in range(total_preds//batch_size):
#         lens, params, noise_rms, psf_fwhm = phys.draw_sersic_batch(batch_size)
#         x_pred, chi = rim.predict(lens, noise_rms, psf_fwhm)
#         # preds are shaped [time steps, batch size, params]
#         x_preds.append(x_pred.numpy())
#         chi_sq.append(chi_sq.numpy())
#         x_true.append(params.numpy())
#     x_preds = np.concatenate(x_preds[-1], axis=0)
#     chi_sq = np.concatenate(chi_sq[-1], axis=0)
#     y_true = np.concatenate(y_true, axis=0)
#     titles = [
#         r"$\theta_E$",
#         r"$\kappa$ axis ratio $q_\kappa$",
#         r"$\kappa$ orientation $\varphi_\kappa$",
#         r"$x_0$",
#         r"$y_0$",
#         r"Shear $\gamma$",
#         r"Shear orientation $\varphi_\gamma$",
#         r"$x_s$",
#         r"$y_s$",
#         r"Source axis ratio $q_s$",
#         r"Source orientation $\varphi_s$",
#         r"Sersic index $n$",
#         r"Half light radius $R_{e}$"
#     ]
#     return fig

if __name__ == '__main__':
    x = np.random.normal(size=[81, 64, 64, 1])
    y = np.random.normal(size=[81, 64, 64, 1])
    fig = reconstruction_plot(x, y)
    fig.savefig("test.png")
    plt.show()
