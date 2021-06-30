from censai.models import CosmosAutoencoder, RayTracer512, UnetModel
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



if __name__ == '__main__':
    test_ray_tracer_512()
    test_resnet_autoencoder()
    test_unet_model()