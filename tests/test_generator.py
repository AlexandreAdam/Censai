from censai.data_generator import Generator, NISGenerator
import tensorflow as tf
import numpy as np

def test_generator():
    gen = Generator()
    for i, (x, y) in enumerate(gen):
        print(i)

    gen = Generator(total_items=100, batch_size=10)
    for i, (x, y) in enumerate(gen):
        print(i)

    gen = Generator(total_items=99, batch_size=10)
    for i, (x, y) in enumerate(gen):
        print(i)


def test_generator_NISGen_rim():
    pixels = 128
    gen = NISGenerator(model="rim", batch_size=1, method="analytic", pixels=pixels)
    gen2 = NISGenerator(method="conv2d", pixels=pixels)
    gen3 = NISGenerator(method="approximate", pixels=pixels)
    for i, (X, source, kap) in enumerate(gen):
        print(i)

    # kappa = tf.random.normal([1, 64, 64, 1])
    # source = tf.random.normal([1, pixels, pixels, 1])
    r_ein = tf.constant(1., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    elp = tf.constant(0.2, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    phi = tf.constant(np.pi/4, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    xlens = tf.constant(0., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    ylens = tf.constant(0, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]

    xs = tf.constant(0., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    ys = tf.constant(0., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    es = tf.constant(0., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    phi_s = tf.constant(0., tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    w_s = tf.constant(0.1, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]

    source = gen.source_model(xs, ys, es, phi_s, w_s)
    im_lensed_a = gen.lens_source(source, r_ein, elp, phi, xlens, ylens)
    im_lensed_c = gen2.lens_source(source, r_ein, elp, phi, xlens, ylens)
    im_lensed_ap = gen3.lens_source(source, r_ein, elp, phi, xlens, ylens)
    kap = gen.kappa_field(xlens, ylens, elp, phi, r_ein)
    alpha = gen.analytical_deflection_angles(xlens, ylens, elp, phi, r_ein)
    alpha_conv = gen.conv2d_deflection_angles(kap)
    alpha_app = gen.approximate_deflection_angles(xlens, ylens, elp, phi, r_ein)

    assert alpha.shape == [1, pixels, pixels, 2]

    return kap, im_lensed_a, im_lensed_c, im_lensed_ap, alpha, alpha_conv, alpha_app

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    k, ima, imc, imap, al, alc, alap = test_generator_NISGen_rim()
    fig, axs = plt.subplots(3, 4, figsize=(16, 8))
    axs[0, 1].imshow(al[0, ..., 0], cmap="hot")
    axs[1, 1].imshow(alc[0, ..., 0], cmap="hot")
    axs[2, 1].imshow(alap[0, ..., 0], cmap="hot")
    axs[0, 2].imshow(al[0, ..., 1], cmap="hot")
    axs[1, 2].imshow(alc[0, ..., 1], cmap="hot")
    axs[2, 2].imshow(alap[0, ..., 1], cmap="hot")

    __im = axs[0, 3].imshow(np.abs(al[0, ..., 0] - alc[0, ..., 0]), cmap="jet")
    divider = make_axes_locatable(axs[0, 3])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(__im, cax=cax)
    __im = axs[1, 3].imshow(np.abs(al[0, ..., 1] - alc[0, ..., 1]), cmap="jet")
    divider = make_axes_locatable(axs[1, 3])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(__im, cax=cax)
    __im = axs[2, 3].imshow(np.abs(al[0, ..., 1] - alc[0, ..., 1]), cmap="jet")
    divider = make_axes_locatable(axs[2, 3])
    cax     = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(__im, cax=cax)
    axs[0, 0].imshow(ima[0, ..., 0], cmap="hot")
    axs[1, 0].imshow(imc[0, ..., 0], cmap="hot")
    axs[2, 0].imshow(imap[0, ..., 0], cmap="hot")
    for i in range(3):
        for j in range(4):
            axs[i, j].axis("off")
    plt.show()

