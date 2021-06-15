from censai.physical_model import PhysicalModel, AnalyticalPhysicalModel
import tensorflow as tf


def test_deflection_angle_conv2():
    phys = PhysicalModel(pixels=64, src_pixels=64)
    kappa = tf.random.normal([1, 64, 64, 1])
    phys.deflection_angle(kappa)


def test_lens_source_conv2():
    pixels = 64
    src_pixels = 32
    phys = PhysicalModel(pixels=pixels, src_pixels=src_pixels, kappa_side=16, image_side=16)
    phys_analytic = AnalyticalPhysicalModel(pixels=pixels, kappa_side=16)
    source = tf.random.normal([1, src_pixels, src_pixels, 1])
    kappa = phys_analytic.kappa_field(7, 0.1, 0, 0, 0)
    lens = phys.lens_source(source, kappa)
    return lens


def test_noisy_forward_conv2():
    phys = PhysicalModel(pixels=64, src_pixels=64)
    source = tf.random.normal([2, 64, 64, 1])
    kappa = tf.math.log(tf.random.uniform([2 , 64, 64, 1], minval=0.01)) / tf.math.log(10.)
    noise_rms = 0.1
    phys.noisy_forward(source, kappa, noise_rms)


def test_log_likelihood():
    phys = PhysicalModel(pixels=64, src_pixels=64)
    kappa = tf.random.normal([1, 64, 64, 1])
    source = tf.random.normal([1, 64, 64, 1])
    im_lensed = phys.forward(source, kappa)
    assert im_lensed.shape == [1, 64, 64, 1]
    cost = phys.log_likelihood(source, kappa, im_lensed)


def test_analytical_lensing():
    phys = AnalyticalPhysicalModel()
    source = tf.random.normal([1, 256, 256, 1])
    params = [1, 0.1, 0, 0.1, -0.1, 0.01, 3.14]
    im = phys.lens_source(source, *params)
    return im.numpy()[0, ..., 0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # im = test_analytical_lensing()
    im = test_lens_source_conv2()[0, ..., 0]
    plt.imshow(im)
    plt.show()

