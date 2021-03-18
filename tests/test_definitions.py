from censai.definitions import *
from censai.physical_model import PhysicalModel
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def test_VAE():
    vae = VAE()
    a = tf.random.normal([256, 256])
    out = vae.encoder(a)
    vae.cost(a)
    vae.draw_image(10)

def test_RayTracer():
    ray_tracer = RayTracer()
    a = tf.random.normal(shape=[1, 256, 256, 1])
    out = ray_tracer.call(a)
    out  = ray_tracer.cost(a, a)

def test_Conv_GRU():
    conv_gru = Conv_GRU(6)
    a = tf.random.normal([1, 256, 256, 1])
    state = tf.random.normal([1, 256, 256, 6])
    out = conv_gru.call(a, state)


def test_Model():
    model = Model(6)
    a = tf.random.normal([1, 256, 256, 1])
    states = tf.random.normal([1, 8, 8, 6])
    grad = tf.random.normal([1, 256, 256, 1])
    model.call(a, states, grad)

def test_GRU_COMPONENT():
    gru = GRU_COMPONENT(6)
    a = tf.random.normal(shape=[1, 256, 256, 1])
    states = tf.random.normal(shape=[1, 256, 256, 6])
    out = gru.call(a, states)


def test_RIM_UNET():
    rim = RIM_UNET((6, 6, 6, 6))
    a = tf.random.normal([1, 256, 256, 1])
    size = [256, 64, 16, 4] # strides of 4
    states = [tf.random.normal([1, i, i, 6]) for i in size]
    grad = tf.random.normal([1, 256, 256, 1])
    rim.call(a, states, grad)

def test_RIM_UNET_CELL():
    phys = PhysicalModel()
    rim = RIM_UNET_CELL(phys, batch_size=11, num_steps=12, num_pixels=256) # State size is not used by the model
    a = tf.random.normal([1, 256, 256, 1])
    size = [256, 64, 16, 4] # strides of 4
    state_size = rim.state_size_list
    states = [tf.random.normal([1, s, s, state_size[i]]) for i, s in enumerate(size)]
    grad = tf.random.normal([1, 256, 256, 1])
    # out = rim.forward_pass(a)
    out = rim(a, states, grad, a, states, grad)


def test_RIM_CELL():
    rim = RIM_CELL(batch_size=1, num_steps=12, num_pixels=256, state_size=6)
    a = tf.random.normal([1, 256, 256, 1])
    states = tf.random.normal([1, 8, 8, 6])
    grad = tf.random.normal(a.shape)
    out = rim(a, states, grad, a, states, grad)
    # out = rim.forward_pass(a) #TODO fix forward pass

def test_SRC_KAPPA_Genrator():
    kappa = SRC_KAPPA_Generator()
    kappa.draw_k_s("test")
