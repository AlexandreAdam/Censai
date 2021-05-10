from censai.ray_tracer import RayTracer512
import tensorflow as tf


def test_ray_tracer_512():
    kappa = tf.random.normal(shape=[10, 512, 512, 1])
    model = RayTracer512()
    out = model(kappa)

if __name__ == '__main__':
    test_ray_tracer_512()