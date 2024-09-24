import logging

import Htool
import matplotlib.pyplot as plt
import numpy as np
from create_geometry import create_random_points_in_disk, create_random_points_in_sphere
from define_generators import CustomGenerator

logging.basicConfig(level=logging.INFO)

# Random geometry
size = 1000
dimension = 3
if dimension == 2:
    coordinates = create_random_points_in_disk(size)
elif dimension == 3:
    coordinates = create_random_points_in_sphere(size)


# Htool parameters
eta = 10
epsilon = 0.1
maximal_leaf_size = 50
number_of_children = 2


# Build clusters
cluster_tree_builder = Htool.ClusterTreeBuilder()
cluster_tree_builder.set_maximal_leaf_size(maximal_leaf_size)
target_cluster: Htool.Cluster = cluster_tree_builder.create_cluster_tree(
    coordinates, number_of_children
)
source_cluster: Htool.Cluster = cluster_tree_builder.create_cluster_tree(
    coordinates, number_of_children
)

# Build generator
generator = CustomGenerator(coordinates, coordinates)

# HMatrix
hmatrix_builder = Htool.HMatrixTreeBuilder(epsilon, eta, "S", "L")
hmatrix: Htool.HMatrix = hmatrix_builder.build(
    generator, target_cluster, source_cluster
)

# HMatrix vector product
dense_in_user_numbering = hmatrix.to_dense_in_user_numbering()
np.random.seed(0)
x = np.random.rand(size)
y = generator.mat_vec(x)
y_dense = dense_in_user_numbering.dot(x)
print(np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense), epsilon)


# Output
print(hmatrix.shape[0], hmatrix.shape[1])
print(hmatrix.get_tree_parameters())
print(hmatrix.get_local_information())
fig = plt.figure()

if dimension == 2:
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
elif dimension == 3:
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4)

ax1.set_title("target cluster at depth 1")
ax2.set_title("target cluster at depth 2")
ax3.set_title("source cluster at depth 1")
ax4.set_title("Hmatrix on rank 0")
Htool.plot(ax1, target_cluster, coordinates, 1)
Htool.plot(ax2, target_cluster, coordinates, 2)
Htool.plot(ax3, source_cluster, coordinates, 1)
Htool.plot(ax4, hmatrix)
plt.show()
