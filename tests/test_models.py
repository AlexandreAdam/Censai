from censai.models import CosmosAutoencoder, RayTracer512, UnetModel, SharedUnetModel, RayTracer, Model, ResnetVAE, ResnetEncoder, VAE, VAESecondStage
from censai.models import SharedResUnetModel, SharedResUnetAtrousModel, SharedMemoryResUnetAtrousModel, SharedUnetModel
from censai import RIMUnet, RIMSharedUnet, PhysicalModel, RIM, RIMSharedResAtrous, RIMSharedMemoryResAtrous, RIMSharedResUnet
from censai.models.layers import ConvGRU, ConvGRUBlock, ConvGRUPlusBlock
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


def test_rim_unet():
    lens = tf.random.normal(shape=[1, 64, 64, 1])
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32)
    m1 = UnetModel()
    m2 = UnetModel()
    rim = RIMUnet(phys, m1, m2, steps=2)
    rim.call(lens)


def test_resnet_autoencoder():
    pixels = 128
    AE = CosmosAutoencoder(pixels, layers=6)
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
    model = SharedUnetModel(filters=32, layers=2, gru_architecture="plus")
    source = tf.random.normal([10, 32, 32, 1])
    source_grad = tf.random.normal([10, 32, 32, 1])
    kappa = tf.random.normal([10, 32, 32, 1])
    kappa_grad = tf.random.normal([10, 32, 32, 1])
    states = model.init_hidden_states(input_pixels=32, batch_size=10)
    model(source, kappa, source_grad, kappa_grad, states)


def test_rim_shared_unet():
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32, method="fft")
    unet = SharedUnetModel()
    rim = RIMSharedUnet(phys, unet, 4)
    lens = tf.random.normal(shape=[1, 64, 64, 1])
    source_series, kappa_series, chi_squared_series = rim.call(lens)

    rim = RIMSharedUnet(phys, unet, 4, kappa_normalize=True, source_link="relu")
    lens = tf.random.normal(shape=[1, 64, 64, 1])
    source_series, kappa_series, chi_squared_series = rim.call(lens)


def test_rim():
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32, method="fft")
    lens = tf.random.normal(shape=[1, 64, 64, 1])
    m1 = Model(filters=2)
    m2 = Model(filters=2)
    rim = RIM(phys, m1, m2, steps=2)
    rim.call(lens)


def test_resnet_vae():
    vae = ResnetVAE(pixels=32, layers=3, latent_size=16)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    vae.cost_function_training(x, 1., 1.)

    vae = ResnetVAE(pixels=32, layers=3, res_blocks_in_layer=[5, 10, 15], conv_layers_per_block=3, latent_size=16, batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("original")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="original", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("bn_after_addition")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="bn_after_addition", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("relu_before_addition")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="relu_before_addition", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("relu_only_pre_activation")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="relu_only_pre_activation", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("full_pre_activation")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="full_pre_activation", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))

    print("full_pre_activation_rescale")
    vae = ResnetVAE(pixels=32, layers=3, res_architecture="full_pre_activation_rescale", batch_norm=True, dropout_rate=0.2)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))


def test_vae():
    vae = VAE(pixels=32, layers=3, latent_size=16, conv_layers=4)
    x = tf.random.normal(shape=(5, 32, 32, 1))
    print(vae.cost_function_training(x, 1., 1.))


def test_resnet_encoder():
    encoder = ResnetEncoder(layers=3, res_blocks_in_layer=2, conv_layers_in_resblock=2)
    x =  tf.random.normal(shape=(5, 32, 32, 1))
    encoder.call(x)
    z = encoder.call_with_skip_connections(x)

    encoder = ResnetEncoder(layers=5, res_blocks_in_layer=[2, 5, 10, 15, 30], conv_layers_in_resblock=3)
    x =  tf.random.normal(shape=(5, 32, 32, 1))
    encoder.call(x)
    z = encoder.call_with_skip_connections(x)
    pass


def test_vae_second_stage():
    vae = VAESecondStage(latent_size=4, output_size=64)
    x = tf.random.normal(shape=[100, 64])
    y = vae(x)
    vae.cost_function(x)


def test_convGRU():
    gru = ConvGRU(filters=32)
    x = tf.random.normal(shape=(1, 8, 8, 32))
    states = tf.random.normal(shape=(1, 8, 8, 32))
    gru.call(x, states)

    gru = ConvGRUBlock(filters=32)
    x = tf.random.normal(shape=(1, 8, 8, 32))
    states = tf.random.normal(shape=(1, 8, 8, 64))
    gru.call(x, states)


def test_rimsharedresunet():
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32, method="fft")
    unet = SharedResUnetModel()
    rim = RIMSharedResUnet(phys, unet, 2)
    lens = tf.random.normal(shape=[1, 64, 64, 1], dtype=tf.float32)
    source_series, kappa_series, chi_squared_series = rim.call(lens)


def test_rimsharedresunetatrous():
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32, method="fft")
    unet = SharedResUnetAtrousModel(dilation_rates=[[1, 4, 8, 32], [1, 2, 4, 6]], pixels=32)
    rim = RIMSharedResAtrous(phys, unet, 2)
    lens = tf.random.normal(shape=[1, 64, 64, 1], dtype=tf.float32)
    source_series, kappa_series, chi_squared_series = rim.call(lens)

def test_rim_shared_memoryresunetatrous():
    phys = PhysicalModel(pixels=64, src_pixels=32, kappa_pixels=32, method="fft")
    unet = SharedMemoryResUnetAtrousModel(dilation_rates=[[1, 4, 8, 32], [1, 2, 4, 6]], pixels=32)
    rim = RIMSharedMemoryResAtrous(phys, unet, 2)
    lens = tf.random.normal(shape=[1, 64, 64, 1], dtype=tf.float32)
    source_series, kappa_series, chi_squared_series = rim.call(lens)


if __name__ == '__main__':
    # test_ray_tracer_512()
    # test_raytracer()
    # test_resnet_autoencoder()
    # test_unet_model()
    # test_shared_unet_model()
    # test_rim_shared_unet()
    # test_rim()
    # test_rim_unet()
    # test_resnet_vae()
    # test_resnet_encoder()
    # test_vae()
    # test_vae_second_stage()
    # test_convGRU()
    # test_rimsharedresunet()
    # test_rimsharedresunetatrous()
    test_rim_shared_memoryresunetatrous()