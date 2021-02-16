from censai.physical_model import PhysicalModel
import tensorflow as tf

def test_deflection_angle_conv2():
    phys = PhysicalModel()
    kappa = tf.random.normal([1, 256, 256, 1])
    phys.deflection_angle(kappa)

def test_lens_source_conv2():
    phys = PhysicalModel()
    source = tf.random.normal([2 , 256, 256, 1])
    x = tf.linspace(-1, 1, [256]) * phys.src_side
    xx, yy = tf.meshgrid(x, x)
    xx = xx[tf.newaxis, ..., tf.newaxis]
    yy = yy[tf.newaxis, ..., tf.newaxis]
    xx = tf.cast(xx, tf.float32)
    yy = tf.cast(yy, tf.float32)
    xx = tf.tile(xx, (2, 1, 1, 1))
    yy = tf.tile(yy, (2, 1, 1, 1))
    phys.lens_source(xx, yy, source)

def test_noisy_forward_conv2():
    phys = PhysicalModel()
    source = tf.random.normal([2, 256, 256, 1])
    kappa = tf.math.log(tf.random.uniform([2 , 256, 256, 1], minval=0.01)) / tf.math.log(10.)
    noise_rms = 0.1
    phys.noisy_forward(source, kappa, noise_rms)

