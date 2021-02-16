from censai.data_generator import Generator

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
