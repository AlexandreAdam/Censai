import numpy as np
from censai.physical_model import PhysicalModel, AnalyticalPhysicalModel
import tensorflow as tf
from censai.utils import raytracer_residual_plot


def test_deflection_angle_conv2():
    phys = PhysicalModel(pixels=64, src_pixels=64)
    kappa = tf.random.normal([1, 64, 64, 1])
    phys.deflection_angle(kappa)


def test_lens_source_conv2():
    pixels = 64
    src_pixels = 32
    phys = PhysicalModel(pixels=pixels, src_pixels=src_pixels, kappa_fov=16, image_fov=16)
    phys_analytic = AnalyticalPhysicalModel(pixels=pixels, kappa_side=16)
    source = tf.random.normal([1, src_pixels, src_pixels, 1])
    kappa = phys_analytic.kappa_field(7, 0.1, 0, 0, 0)
    lens = phys.lens_source(source, kappa)
    return lens


def test_alpha_method_fft():
    pixels = 64
    phys = PhysicalModel(pixels=pixels, method="fft")
    phys_analytic = AnalyticalPhysicalModel(pixels=pixels, kappa_side=7)
    phys2 = PhysicalModel(pixels=pixels, method="conv2d")

    # test out noise
    kappa = tf.random.uniform(shape=[1, pixels, pixels, 1])
    _, _, alphax, alphay = phys.deflection_angle(kappa)
    _, _, alphax2, alphay2 = phys2.deflection_angle(kappa)

    assert np.allclose(alphax, alphax2, atol=1e-4)
    assert np.allclose(alphay, alphay2, atol=1e-4)

    # test out an analytical profile
    kappa = phys_analytic.kappa_field(2, 0., 0, 0.1, 0.5)
    _, _, alphax, alphay = phys.deflection_angle(kappa)

    _, _, alphax2, alphay2 = phys2.deflection_angle(kappa)

    assert np.allclose(alphax, alphax2, atol=1e-4)
    assert np.allclose(alphay, alphay2, atol=1e-4)

    return alphax, alphax2


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
    params = [1., 0.1, 0., 0.1, -0.1, 0.01, 3.14]
    im = phys.lens_source(source, *params)

    im = phys.lens_source_func(e=0.6)

    kap = phys.kappa_field(e=0.2)
    return im.numpy()[0, ..., 0]


def test_lens_func_given_alpha():
    phys = PhysicalModel(pixels=128)
    phys_a = AnalyticalPhysicalModel(pixels=128)
    alpha = phys_a.approximate_deflection_angles(x0=0.5, y0=0.5, e=0., phi=0., r_ein=1.)
    lens_true = phys_a.lens_source_func(x0=0.5, y0=0.5, xs=0.5, ys=0.5)
    lens_pred = phys_a.lens_source_func_given_alpha(alpha, xs=0.5, ys=0.5)
    lens_pred2 = phys.lens_source_func_given_alpha(alpha, xs=0.5, ys=0.5)
    fig = raytracer_residual_plot(alpha[0], alpha[0], lens_true[0], lens_pred[0])
    cost1 = tf.reduce_mean(tf.cast(tf.math.equal(lens_true, lens_pred), tf.float32))
    cost2 = tf.reduce_mean(tf.cast(tf.math.equal(lens_true, lens_pred2), tf.float32))
    # assert cost1 == 1.0, cost1
    # assert cost2 == 1.0, cost2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    test_lens_func_given_alpha()
    plt.show()
    im = test_analytical_lensing()
    # im = test_lens_source_conv2()[0, ..., 0]
    im1, im2 = test_alpha_method_fft()
    plt.imshow(im)
    plt.colorbar()
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4))

    im = ax1.imshow(im1[0, ..., 0])
    ax1.set_title("FFT")
    ax1.axis("off")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = ax2.imshow(im2[0, ..., 0])
    ax2.set_title("Conv")
    ax2.axis("off")
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = ax3.imshow(im2[0, ..., 0] - im1[0, ..., 0])
    ax3.set_title("Residual")
    ax3.axis("off")
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    plt.show()


