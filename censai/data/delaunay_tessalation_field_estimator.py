import numpy as np
from scipy.spatial import Delaunay  # careful to choose the proper QHull method to run this at O(n log n)


def tetrahedron_volume(simplices):
    # shape is [batch_size, xyz, 4 vertex]
    DA = simplices[..., 0] - simplices[..., 3] # A - D
    DB = simplices[..., 1] - simplices[..., 3] # B - D
    DC = simplices[..., 2] - simplices[..., 3] # C - D
    return np.abs(np.einsum("...i, ...i", DA, np.cross(DB, DC))) / 6


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    start = time.time()
    x = np.random.normal(size=(1000, 2))
    tri = Delaunay(x) # 2 seconds for 100k, 25 seconds for 1M. Should be good for the few tens of millions in a single halo.
    print(time.time() - start)
    plt.triplot(x[:,0], x[:,1], tri.simplices)
    plt.plot(x[:,0], x[:,1], 'o')
    plt.show()