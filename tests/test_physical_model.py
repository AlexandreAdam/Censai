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
    phys_analytic = AnalyticalPhysicalModel(pixels=pixels, image_fov=16)
    source = tf.random.normal([1, src_pixels, src_pixels, 1])
    kappa = phys_analytic.kappa_field(7, 0.1, 0, 0, 0)
    lens = phys.lens_source(source, kappa)
    return lens


def test_alpha_method_fft():
    pixels = 64
    phys = PhysicalModel(pixels=pixels, method="fft")
    phys_analytic = AnalyticalPhysicalModel(pixels=pixels, image_fov=7)
    phys2 = PhysicalModel(pixels=pixels, method="conv2d")

    # test out noise
    kappa = tf.random.uniform(shape=[1, pixels, pixels, 1])
    alphax, alphay = phys.deflection_angle(kappa)
    alphax2, alphay2 = phys2.deflection_angle(kappa)

    assert np.allclose(alphax, alphax2, atol=1e-4)
    assert np.allclose(alphay, alphay2, atol=1e-4)

    # test out an analytical profile
    kappa = phys_analytic.kappa_field(2, 0.4, 0, 0.1, 0.5)
    alphax, alphay = phys.deflection_angle(kappa)

    alphax2, alphay2 = phys2.deflection_angle(kappa)

    assert np.allclose(alphax, alphax2, atol=1e-4)
    assert np.allclose(alphay, alphay2, atol=1e-4)

    return alphax, alphax2


def test_noisy_forward_conv2():
    phys = PhysicalModel(pixels=64, src_pixels=64)
    source = tf.random.normal([2, 64, 64, 1])
    kappa = tf.math.exp(tf.random.uniform([2 , 64, 64, 1]))
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
    alpha = phys_a.analytical_deflection_angles(x0=0.5, y0=0.5, e=0.4, phi=0., r_ein=1.)
    lens_true = phys_a.lens_source_func(x0=0.5, y0=0.5, e=0.4, phi=0., r_ein=1., xs=0.5, ys=0.5)
    lens_pred = phys_a.lens_source_func_given_alpha(alpha, xs=0.5, ys=0.5)
    lens_pred2 = phys.lens_source_func_given_alpha(alpha, xs=0.5, ys=0.5)
    fig = raytracer_residual_plot(alpha[0], alpha[0], lens_true[0], lens_pred2[0])
    # assert np.allclose(lens_pred2, lens_true, atol=1e-5)
    # assert np.allclose(lens_pred, lens_true, atol=1e-5)


def test_interpolated_kappa():
    import tensorflow_addons as tfa
    phys = PhysicalModel(pixels=128, src_pixels=32, image_fov=7.68, kappa_fov=5)
    phys_a = AnalyticalPhysicalModel(pixels=128, image_fov=7.68)
    kappa = phys_a.kappa_field(r_ein=2., e=0.2)
    kappa += phys_a.kappa_field(r_ein=1., x0=2., y0=2.)
    true_lens = phys.lens_source_func(kappa, w=0.2)
    true_kappa = kappa

    # Test interpolation of alpha angles on a finer grid
    # phys = PhysicalModel(pixels=128, src_pixels=32, kappa_pixels=32)
    phys_a = AnalyticalPhysicalModel(pixels=32, image_fov=7.68)
    kappa = phys_a.kappa_field(r_ein=2., e=0.2)
    kappa += phys_a.kappa_field(r_ein=1., x0=2., y0=2.)

    # kappa2 = phys_a.kappa_field(r_ein=2., e=0.2)
    # kappa2 += phys_a.kappa_field(r_ein=1., x0=2., y0=2.)
    #
    # kappa = tf.concat([kappa, kappa2], axis=1)

    # Test interpolated kappa lens
    x = np.linspace(-1, 1, 128) * phys.kappa_fov / 2
    x, y = np.meshgrid(x, x)
    x = tf.constant(x[np.newaxis, ..., np.newaxis], tf.float32)
    y = tf.constant(y[np.newaxis, ..., np.newaxis], tf.float32)
    dx = phys.kappa_fov / (32 - 1)
    xmin = -0.5 * phys.kappa_fov
    ymin = -0.5 * phys.kappa_fov
    i_coord = (x - xmin) / dx
    j_coord = (y - ymin) / dx
    wrap = tf.concat([i_coord, j_coord], axis=-1)
    # test_kappa1 = tfa.image.resampler(kappa, wrap)  # bilinear interpolation of source on wrap grid
    # test_lens1 = phys.lens_source_func(test_kappa1, w=0.2)
    phys2 = PhysicalModel(pixels=128, kappa_pixels=32, method="fft", image_fov=7.68, kappa_fov=5)
    test_lens1 = phys2.lens_source_func(kappa, w=0.2)

    # Test interpolated alpha angles lens
    phys2 = PhysicalModel(pixels=32, src_pixels=32, image_fov=7.68, kappa_fov=5)
    alpha1, alpha2 = phys2.deflection_angle(kappa)
    alpha = tf.concat([alpha1, alpha2], axis=-1)
    alpha = tfa.image.resampler(alpha, wrap)
    test_lens2 = phys.lens_source_func_given_alpha(alpha, w=0.2)

    return true_lens, test_lens1, test_lens2


def test_lagrange_multiplier_for_lens_intensity():
    phys = PhysicalModel(pixels=128)
    phys_a = AnalyticalPhysicalModel(pixels=128)
    kappa = phys_a.kappa_field(2.0, e=0.2)
    x = np.linspace(-1, 1, 128) * phys.src_fov/2
    xx, yy = np.meshgrid(x, x)
    rho = xx**2 + yy**2
    source = tf.math.exp(-0.5 * rho / 0.5**2)[tf.newaxis, ..., tf.newaxis]
    source = tf.cast(source, tf.float32)

    y_true = phys.forward(source, kappa)
    y_pred = phys.forward(0.001 * source, kappa)  # rescale it, say it has different units
    lam_lagrange = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)) / tf.reduce_sum(y_pred ** 2, axis=(1, 2, 3))
    lam_tests = tf.squeeze(tf.cast(tf.linspace(lam_lagrange/10, lam_lagrange*10, 1000), tf.float32))[..., tf.newaxis, tf.newaxis, tf.newaxis]
    log_likelihood_best = 0.5 * tf.reduce_mean((lam_lagrange * y_pred - y_true) ** 2 / phys.noise_rms ** 2, axis=(1, 2, 3))
    log_likilhood_test = 0.5 * tf.reduce_mean((lam_tests * y_pred - y_true) ** 2 / phys.noise_rms ** 2, axis=(1, 2, 3))
    return log_likilhood_test, log_likelihood_best, tf.squeeze(lam_tests), lam_lagrange


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm, SymLogNorm, CenteredNorm
    test_lens_func_given_alpha()
    im = test_analytical_lensing()
    # im = test_lens_source_conv2()[0, ..., 0]
    im1, im2 = test_alpha_method_fft()
    plt.imshow(im)
    plt.colorbar()

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

    true_lens, test_lens1, test_lens2 = test_interpolated_kappa()
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(true_lens[0, ..., 0], cmap="hot")
    axs[0, 0].axis("off")
    axs[0, 0].set_title("Ground Truth")
    axs[0, 1].imshow(test_lens1[0, ..., 0], cmap="hot")
    axs[0, 1].axis("off")
    axs[0, 1].set_title("Kappa interpolation")
    axs[1, 1].imshow(test_lens1[0, ..., 0], cmap="hot")
    axs[1, 1].axis("off")
    axs[1, 1].set_title("Alpha interpolation")
    # axs[1, 0].imshow(true_lens[0, ..., 0], cmap="hot")
    axs[1, 0].axis("off")
    axs[0, 2].imshow(true_lens[0, ..., 0] - test_lens1[0, ..., 0], cmap="seismic", norm=CenteredNorm())
    axs[0, 2].set_title("Residuals")
    axs[0, 2].axis("off")
    axs[1, 2].imshow(true_lens[0, ..., 0] - test_lens2[0, ..., 0], cmap="seismic", norm=CenteredNorm())
    axs[1, 2].axis("off")

    log_test, log_best, lam_test, lam_best = test_lagrange_multiplier_for_lens_intensity()
    plt.figure()
    plt.plot(lam_test, log_test, "-k")
    plt.axvline(lam_best)
    plt.axhline(log_best)
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\mathcal{L}(y \mid x)$")

    plt.show()
