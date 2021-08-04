import tensorflow as tf
import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

COSMO = cosmo
DTYPE = tf.float32
LOG10 = tf.constant(np.log(10.), DTYPE)
LOGFLOOR = tf.constant(1e-8, DTYPE)
# some estimate of kappa statistics (after rescaling for theta_e ~ Uniform(1, 7))
KAPPA_LOG_MEAN = -0.52
KAPPA_LOG_STD = 0.3
KAPPA_LOG_MAX = 3
KAPPA_LOG_MIN = tf.math.log(LOGFLOOR) / tf.math.log(10.)  # not actual min, which is 0

SIGMOID_MIN = tf.constant(1e-3, DTYPE)
SIGMOIN_MAX = tf.constant(1 - 1e-3, DTYPE)


def theta_einstein(kappa, rescaling, physical_pixel_scale, sigma_crit, Dds, Ds, Dd):
    """ #TODO deal with astropy units internally in this function
    Einstein radius is computed with the mass inside the Einstein ring, which corresponds to
    where kappa > 1.
    Args:
        kappa: A single kappa map of shape [crop_pixels, crop_pixels, channels]
        rescaling: Possibly an array of rescaling factor of the kappa maps
        physical_pixel_scale: Pixel scale in Comoving Mpc for the kappa map grid (in Mpc/pixels, with astropy units)
        sigma_crit: Critical Density (should be in units kg / Mpc**2 with astropy units attached)
        Dds: Angular diameter distance between the deflector (d) and the source (s) (in Mpc with astropy units attached)
        Ds: Augular diameter distance between the observer and the source (in Mpc with astropy units attached)
        Dd: Augular diameter distance between the observer and the deflector (in Mpc with astropy units attached)

    Returns: Einstein radius in arcsecond (an array of floats, no astropy units attached)
    """
    rescaling = np.atleast_1d(rescaling)
    kap = rescaling[..., np.newaxis, np.newaxis, np.newaxis] * kappa[np.newaxis, ...]
    mass_inside_einstein_radius = np.sum(kap * (kap > 1), axis=(1, 2, 3)) * sigma_crit * physical_pixel_scale**2
    return (np.sqrt(4 * G / c ** 2 * mass_inside_einstein_radius * Dds / Ds / Dd).decompose() * u.rad).to(u.arcsec).value


def compute_rescaling_probabilities(kappa, rescaling_array, physical_pixel_scale, sigma_crit, Dds, Ds, Dd,
                                    bins=10, min_theta_e=1., max_theta_e=5.,
                                    ):
    """
    Args:
        kappa: A single kappa map, of shape [crop_pixels, crop_pixels, channel]
        rescaling_array: An array of rescaling factor
        bins: Number of bins of the histogram used to figure out einstein radius distribution
        min_theta_e: Minimum desired value of Einstein ring radius (in arcsec)
        max_theta_e: Maximum desired value of Einstein ring radius (in arcsec)

    Returns: Probability of picking rescaling factor in rescaling array so that einstein radius has a
        uniform distribution between minimum and maximum allowed value
    """
    p = np.zeros_like(rescaling_array)
    theta_e = theta_einstein(kappa, rescaling_array, physical_pixel_scale, sigma_crit, Dds, Ds, Dd)
    # compute theta distribution
    select = (theta_e >= min_theta_e) & (theta_e <= max_theta_e)
    if select.sum() == 0:
        return p
    theta_hist, bin_edges = np.histogram(theta_e, bins=bins, range=[min_theta_e, max_theta_e], density=False)
    # for each theta_e, find bin index of our histogram. We give the left edges of the bin (param right=False)
    rescaling_bin = np.digitize(theta_e[select], bin_edges[:-1],
                                right=False) - 1  # bin 0 is outside the range to the left by default
    theta_hist[theta_hist == 0] = 1  # give empty bins a weight
    p[select] = 1 / theta_hist[rescaling_bin]
    p /= p.sum()  # normalize our new probability distribution
    return p


@tf.function
def bipolar_elu(x):
    """Bipolar ELU as in https://arxiv.org/abs/1709.04054."""
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.elu(x1)
    y2 = -tf.nn.elu(-x2)
    return tf.concat([y1, y2], axis=-1)


@tf.function
def bipolar_leaky_relu(x, alpha=0.2, **kwargs):
    """Bipolar Leaky ReLU as in https://arxiv.org/abs/1709.04054."""
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.leaky_relu(x1, alpha=alpha)
    y2 = -tf.nn.leaky_relu(-x2, alpha=alpha)
    return tf.concat([y1, y2], axis=-1)


@tf.function
def bipolar_relu(x):
    x1, x2 = tf.split(x, 2, axis=-1)
    y1 = tf.nn.relu(x1)
    y2 = -tf.nn.relu(-x2)
    return tf.concat([y1, y2], axis=-1)


def lrelu(x, alpha=0.3):
    return tf.maximum(x, tf.multiply(x, alpha))


def endlrelu(x, alpha=0.06):
    return tf.maximum(x, tf.multiply(x, alpha))


def m_softplus(x):
    return tf.keras.activations.softplus(x) - tf.keras.activations.softplus( -x -5.0 )


def xsquared(x):
    return (x/4)**2


def log_10(x):
    return tf.math.log(x + LOGFLOOR) / LOG10


def kappa_clipped_exponential(log_kappa):
    # log_kappa = tf.clip_by_value(log_kappa, clip_value_min=KAPPA_LOG_MIN, clip_value_max=KAPPA_LOG_MAX)
    # clip values of log_kappa in a certain range. Make sure sure it is differentiable
    log_kappa = (log_kappa - KAPPA_LOG_MIN) * 4 / (KAPPA_LOG_MAX - KAPPA_LOG_MIN) - 2  # rescale between -2 and 2 for tanh
    log_kappa = (tf.math.tanh(log_kappa) + 1) * (KAPPA_LOG_MAX - KAPPA_LOG_MIN) / 2 + KAPPA_LOG_MIN  # rescale output of tanh to wanted range
    return 10**log_kappa


def logkappa_normalization(x, forward=True):
    if forward:
        return (x - KAPPA_LOG_MEAN) / KAPPA_LOG_STD
    else:
        return KAPPA_LOG_STD * x + KAPPA_LOG_MEAN


def lrelu4p(x, alpha=0.04):
    return tf.maximum(x, tf.multiply(x, alpha))


def logit(x):
    """
    Computes the logit function, i.e. the logistic sigmoid inverse.
    This function has no gradient, so it cannot be used on model output. It is mainly used
    to link physical labels to model labels.

    We clip values for numerical stability.
    Normally, there should not be any values outside of this range anyway, except maybe for the peak at 1.
    """
    x = tf.math.minimum(x, SIGMOIN_MAX)
    x = tf.math.maximum(x, SIGMOID_MIN)
    return - tf.math.log(1. / x - 1.)


def to_float(x):
    """Cast x to float; created because tf.to_float is deprecated."""
    return tf.cast(x, tf.float32)


def inverse_exp_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay exponentially from min_value to 1.0 reached at max_step."""
    inv_base = tf.exp(tf.math.log(min_value) / float(max_step))
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    return inv_base**tf.maximum(float(max_step) - step, 0.0)


class PolynomialSchedule:
    def __init__(self, initial_value, end_value, power, decay_steps, cyclical=False):
        self.initial_value = initial_value
        self.end_value = end_value
        self.power = power
        self.decay_steps = decay_steps
        self.cyclical = cyclical

    def __call__(self, step=None):
        if step is None:
            step = tf.summary.experimental.get_step()
        if self.cyclical:
            step = min(step % (2 * self.decay_steps), self.decay_steps)
        else:
            step = min(step, self.decay_steps)
        return ((self.initial_value - self.end_value) * (1 - step / self.decay_steps) ** (self.power)) + self.end_value


def inverse_lin_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)
    progress = tf.minimum(step / float(max_step), 1.0)
    return progress * (1.0 - min_value) + min_value


def inverse_sigmoid_decay(max_step, min_value=0.01, step=None):
    """Inverse-decay linearly from min_value to 1.0 reached at max_step."""
    if step is None:
        step = tf.summary.experimental.get_step()
    if step is None:
        return 1.0
    step = to_float(step)

    def sigmoid(x):
        return 1 / (1 + tf.exp(-x))

    def inv_sigmoid(y):
        return tf.math.log(y / (1 - y))

    assert min_value > 0, (
          "sigmoid's output is always >0 and <1. min_value must respect "
          "these bounds for interpolation to work.")
    assert min_value < 0.5, "Must choose min_value on the left half of sigmoid."

    # Find
    #   x  s.t. sigmoid(x ) = y_min and
    #   x' s.t. sigmoid(x') = y_max
    # We will map [0, max_step] to [x_min, x_max].
    y_min = min_value
    y_max = 1.0 - min_value
    x_min = inv_sigmoid(y_min)
    x_max = inv_sigmoid(y_max)

    x = tf.minimum(step / float(max_step), 1.0)  # [0, 1]
    x = x_min + (x_max - x_min) * x  # [x_min, x_max]
    y = sigmoid(x)  # [y_min, y_max]

    y = (y - y_min) / (y_max - y_min)  # [0, 1]
    y = y * (1.0 - y_min)  # [0, 1-y_min]
    y += y_min  # [y_min, 1]
    return y


def conv2_layers_flops(layer):
    _, _, _, input_channels = layer.input_shape
    _, h, w, output_channels = layer.output_shape
    w_h, w_w = layer.kernel_size
    strides_h, strides_w = layer.strides
    flops = h * w * input_channels * output_channels * w_h * w_w / strides_w / strides_h

    flops_bias = np.prod(layer.output_shape[1:]) if layer.use_bias else 0
    flops = 2 * flops + flops_bias  # times 2 since we must consider multiplications and additions
    return flops


def upsampling2d_layers_flops(layer):
    _, h, w, output_channels = layer.output_shape
    return 50 * h * w * output_channels
