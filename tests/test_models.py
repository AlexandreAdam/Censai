from censai.models.ray_tracer512 import RayTracer512
import tensorflow as tf


def test_ray_tracer_512():
    kappa = tf.random.normal(shape=[10, 512, 512, 1])
    model = RayTracer512(initializer="glorot_uniform", bottleneck_kernel_size=8, bottleneck_strides=2, upsampling_interpolation=True, decoder_encoder_filters=16,
                         batch_norm=True, activation="gelu")
    out = model(kappa)

if __name__ == '__main__':
    test_ray_tracer_512()