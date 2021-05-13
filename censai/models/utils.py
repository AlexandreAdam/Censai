import tensorflow as tf


def summary_histograms(layer, activation):
    tf.summary.histogram(layer.name + "_activation", data=activation, step=tf.summary.experimental.get_step())
    for weights in layer.trainable_weights:
        tf.summary.histogram(weights.name, data=weights, step=tf.summary.experimental.get_step())


def global_step():
    step = tf.summary.experimental.get_step()
    if step is not None:
        return step
    else:
        return -1


def get_activation(activation_name, **kwargs):
    if activation_name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(**kwargs)
    elif activation_name == "gelu":
        return tf.keras.activations.gelu
    else:
        return tf.keras.layers.Activation(activation_name, **kwargs)
