import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import euclidean, cityblock

# Points as given in the image
points = np.array([
    [0.780, -0.039, 0.006, -0.091],
    [0.385, 0.088, 0.687, -0.253],
    [0.696, -0.790, -0.855, 1.195],
    [-0.113, 1.423, -1.075, 0.606]
])

# Calculate Euclidean distance matrix
distance_matrix = squareform(pdist(points, metric='euclidean'))
print(distance_matrix)

point_a = (0.780, -0.039, 0.006, -0.091)
point_b = (0.696, -0.790, -0.855, 1.195)

# Euclidean distance
euclidean_dist = euclidean(point_a, point_b)
print(f"Euclidean distance: {euclidean_dist}")

