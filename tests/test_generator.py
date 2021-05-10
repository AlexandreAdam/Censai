from censai.data_generator import Generator, NISGenerator
import tensorflow as tf

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
    gen = NISGenerator(model="rim", batch_size=1)
    for i, (X, source, kap) in enumerate(gen):
        print(i)

    # kappa = tf.random.normal([1, 64, 64, 1])
    source = tf.random.normal([1, 128, 128, 1])
    r_ein = tf.constant(1.5, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    elp = tf.constant(0.1, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    phi = tf.constant(2, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    xlens = tf.constant(0.5, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]
    ylens = tf.constant(0, tf.float32)[tf.newaxis, tf.newaxis, tf.newaxis]

    im_lensed = gen.lens_source(source, r_ein, elp, phi, xlens, ylens)
    alpha = gen.deflection_angles(xlens, ylens, elp, phi, r_ein)

    return X, source, kap, im_lensed, alpha

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x, s, k, im, al = test_generator_NISGen_rim()
    plt.imshow(al.numpy()[0, ..., 0], cmap="hot")
    plt.colorbar()
    plt.show()

