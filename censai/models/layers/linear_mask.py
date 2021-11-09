import tensorflow as tf


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [tf.range(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [tf.range(hidden_size) % (input_size - 1)]
        degrees += [tf.range(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [tf.random.shuffle(tf.range(input_size))] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [tf.random.uniform(minval=min_prev_degree, maxval=input_size, size=(hidden_size,), dtype=tf.int32)]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [tf.random.uniform(minval=min_prev_degree, maxval=input_size, size=(input_size,), dtype=tf.int32) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [tf.cast(d1[..., None] >= d0[None, ...], tf.float32)]

    return masks, degrees[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mask, degrees = create_masks(5, 32, 2, input_order='sequential', input_degrees=None)
    print(mask)
