from censai.models import CosmosAutoencoder, RayTracer512, UnetModel, SharedUnetModel, RayTracer
from censai import RIMUnet, RIMSharedUnet, RIMUnet512, PhysicalModel
import tensorflow as tf


def test_ray_tracer_512():
    kappa = tf.random.normal(shape=[10, 512, 512, 1])
    model = RayTracer512(initializer="random_uniform",
                         bottleneck_kernel_size=8,
                         bottleneck_strides=2,
                         upsampling_interpolation=True,
                         decoder_encoder_filters=16,
                         activation="linear")
    out = model(kappa)


def test_resnet_autoencoder():
    pixels = 128
    AE = CosmosAutoencoder(pixels)
    image = tf.random.uniform(shape=[1, 128, 128, 1])
    psf = tf.abs(tf.signal.rfft2d(tf.random.normal(shape=[1, 256, 256]))[..., tf.newaxis])
    ps = tf.abs(tf.signal.rfft2d(tf.random.normal(shape=[1, 128, 128]))[..., tf.newaxis])
    AE(image, image)
    train_cost = AE.training_cost_function(image, psf, ps,
                                           skip_strength=1,
                                           l2_bottleneck=1,
                                           apodization_alpha=0.5,
                                           apodization_factor=1,
                                           tv_factor=1
                    )


def test_raytracer():
    model = RayTracer(
        pixels=128,
        filters=32
    )
    X = tf.random.uniform(shape=[10, 128, 128, 1])
    alpha = model(X)
    assert alpha.shape[-1] == 2

    model = RayTracer(
        pixels=128,
        layers=6,
        filters=32
    )
    alpha = model(X)


def test_unet_model():
    # test out the plumbing
    model = UnetModel(filters=32, layers=1)
    X = tf.random.normal([10, 128, 128, 1])
    grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=128, batch_size=10)
    model(X, states, grad)

    model = UnetModel(filters=32, layers=2)
    X = tf.random.normal([10, 128, 128, 1])
    grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=128, batch_size=10)
    model(X, states, grad)

    model = UnetModel(filters=32, bottleneck_filters=32, filter_scaling=2)
    X = tf.random.normal([10, 128, 128, 1])
    grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=128, batch_size=10)
    model(X, states, grad)

    model = UnetModel(filters=4, bottleneck_filters=32, filter_scaling=2, resampling_kernel_size=5, gru_kernel_size=5, upsampling_interpolation=True)
    X = tf.random.normal([10, 128, 128, 1])
    grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=128, batch_size=10)
    model(X, states, grad)

    model = UnetModel(filters=4, layers=6)
    X = tf.random.normal([10, 128, 128, 1])
    grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=128, batch_size=10)
    model(X, states, grad)


def test_shared_unet_model():
    # test out plumbing
    model = SharedUnetModel(kappa_resize_layers=2, filters=32, layers=1)
    source = tf.random.normal([10, 32, 32, 1])
    source_grad = tf.random.normal([10, 32, 32, 1])
    kappa = tf.random.normal([10, 128, 128, 1])
    kappa_grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=32, batch_size=10)
    model(source, kappa, source_grad, kappa_grad, states)

    # test out plumbing
    model = SharedUnetModel(kappa_resize_layers=2, filters=32, layers=2)
    source = tf.random.normal([10, 32, 32, 1])
    source_grad = tf.random.normal([10, 32, 32, 1])
    kappa = tf.random.normal([10, 128, 128, 1])
    kappa_grad = tf.random.normal([10, 128, 128, 1])
    states = model.init_hidden_states(input_pixels=32, batch_size=10)
    model(source, kappa, source_grad, kappa_grad, states)

    # test out plumbing
    model = SharedUnetModel(kappa_resize_layers=0, filters=32, layers=2)
    source = tf.random.normal([10, 32, 32, 1])
    source_grad = tf.random.normal([10, 32, 32, 1])
    kappa = tf.random.normal([10, 32, 32, 1])
    kappa_grad = tf.random.normal([10, 32, 32, 1])
    states = model.init_hidden_states(input_pixels=32, batch_size=10)
    model(source, kappa, source_grad, kappa_grad, states)

    # test out plumbing
    model = SharedUnetModel(kappa_resize_layers=1, filters=32, layers=2)
    source = tf.random.normal([10, 32, 32, 1])
    source_grad = tf.random.normal([10, 32, 32, 1])
    kappa = tf.random.normal([10, 64, 64, 1])
    kappa_grad = tf.random.normal([10, 64, 64, 1])
    states = model.init_hidden_states(input_pixels=32, batch_size=10)
    model(source, kappa, source_grad, kappa_grad, states)


def test_rim_shared_unet():  # TODO check that no output is nan
    phys = PhysicalModel(pixels=64, src_pixels=32, method="fft")
    unet = SharedUnetModel(kappa_resize_layers=1)
    rim = RIMSharedUnet(phys, unet, 4)
    lens = tf.random.normal(shape=[1, 64, 64, 1])
    source_series, kappa_series, chi_squared_series = rim.call(lens)

    phys = PhysicalModel(pixels=32, src_pixels=32, method="fft")
    unet = SharedUnetModel(kappa_resize_layers=0)
    rim = RIMSharedUnet(phys, unet, 4, kappa_normalize=True, source_link="sqrt")
    lens = tf.random.normal(shape=[1, 32, 32, 1])
    source_series, kappa_series, chi_squared_series = rim.call(lens)

    rim = RIMSharedUnet(phys, unet, 4, kappa_normalize=True, source_link="exp")
    lens = tf.random.normal(shape=[1, 32, 32, 1])
    source_series, kappa_series, chi_squared_series = rim.call(lens)


if __name__ == '__main__':
    test_ray_tracer_512()
    test_raytracer()
    test_resnet_autoencoder()
    test_unet_model()
    test_shared_unet_model()
    test_rim_shared_unet()