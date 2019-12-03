from scipy.spatial import distance as dist
import numpy as np
np.random.seed(42)
objectCentroids = np.random.uniform(size=(2, 2))
print(objectCentroids)
centroids = np.random.uniform(size=(3, 2))
print(centroids)
D = dist.cdist(objectCentroids, centroids)
print(D)
print(D.min(axis=1))