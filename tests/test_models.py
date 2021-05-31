from censai.models import Autoencoder, RayTracer512
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
    AE = Autoencoder(pixels)
    image = tf.random.uniform(shape=[1, 128, 128, 1])
    psf = tf.abs(tf.signal.rfft2d(tf.random.normal(shape=[1, 256, 256]))[..., tf.newaxis])
    ps = tf.abs(tf.signal.rfft2d(tf.random.normal(shape=[1, 128, 128]))[..., tf.newaxis])
    train_cost = AE.training_cost_function(image, psf, ps,
                                           skip_strength=1,
                                           l2_bottleneck=1,
                                           apodization_alpha=0.5,
                                           apodization_factor=1,
                                           tv_factor=1
                    )

if __name__ == '__main__':
    # test_ray_tracer_512()
    test_resnet_autoencoder()