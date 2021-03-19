from censai.physical_model import PhysicalModel, AnalyticalPhysicalModel
import tensorflow as tf


def test_deflection_angle_conv2():
    phys = PhysicalModel(pixels=64)
    kappa = tf.random.normal([1, 64, 64, 1])
    phys.deflection_angle(kappa)


def test_lens_source_conv2():
    phys = PhysicalModel(pixels=64)
    source = tf.random.normal([2 , 64, 64, 1])
    x = tf.linspace(-1, 1, [64]) * phys.src_side
    xx, yy = tf.meshgrid(x, x)
    xx = xx[tf.newaxis, ..., tf.newaxis]
    yy = yy[tf.newaxis, ..., tf.newaxis]
    xx = tf.cast(xx, tf.float32)
    yy = tf.cast(yy, tf.float32)
    xx = tf.tile(xx, (2, 1, 1, 1))
    yy = tf.tile(yy, (2, 1, 1, 1))
    phys.lens_source(xx, yy, source)


def test_noisy_forward_conv2():
    phys = PhysicalModel(pixels=64)
    source = tf.random.normal([2, 64, 64, 1])
    kappa = tf.math.log(tf.random.uniform([2 , 64, 64, 1], minval=0.01)) / tf.math.log(10.)
    noise_rms = 0.1
    phys.noisy_forward(source, kappa, noise_rms)


def test_log_likelihood():
    phys = PhysicalModel(pixels=64)
    kappa = tf.random.normal([1, 64, 64, 1])
    source = tf.random.normal([1, 64, 64, 1])
    im_lensed = phys.forward(source, kappa)
    assert im_lensed.shape == [1, 64, 64, 1]
    cost = phys.log_likelihood(source, kappa, im_lensed)

def test_analytical_lensing():
    phys = AnalyticalPhysicalModel()
    source = tf.random.normal([1, 256, 256, 1])
    # x = tf.linspace(-1, 1, 256) * 3 # as
    # xx, yy = tf.meshgrid(x, x)
    # xs = 0.1
    # ys = -0.1
    # rho = tf.sqrt((xx - xs)**2 + (yy - ys)**2)
    # source = tf.exp(-rho**2/0.2**2)
    # source = tf.cast(source, tf.float32)
    # source = source[tf.newaxis, ..., tf.newaxis]
    params = [1, 0.1, 0, 0.1, -0.1, 0.01, 3.14]
    im = phys.lens_source(source, *params)
    return im.numpy()[0, ..., 0]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    im = test_analytical_lensing()
    plt.imshow(im)
    plt.show()

