import numpy as np
from scipy.spatial import Delaunay


# def tetrahedron_volume(simplices):
#     # shape is [batch_size, xyz, vertex]
#     return np.abs(np.dot())


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.random.normal(size=(1000, 2))
    tri = Delaunay(x)
    plt.triplot(x[:,0], x[:,1], tri.simplices)
    plt.plot(x[:,0], x[:,1], 'o')
    plt.show()