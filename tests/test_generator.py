from censai.data_generator import Generator, NISGenerator

def test_generator():
    gen = Generator()
    for i, (x, y) in enumerate(gen):
        print(i)

    gen = Generator(total_items=100, batch_size=10)
    for i, (x, y) in enumerate(gen):
        print(i)

    gen = Generator(total_items=99, batch_size=10)
    for i, (x, y) in enumerate(gen):
        print(i)


def test_generator_NISGen_rim():
    gen = NISGenerator(model="rim", batch_size=2)
    for i, (kap, source, Y) in enumerate(gen):
        print(i)

    print(kap.shape)
    print(source.shape)
    print(Y.shape)
    return kap, source, Y

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    k, s, y = test_generator_NISGen_rim()
    plt.imshow(y.numpy()[0, ..., 0])
    plt.show()

