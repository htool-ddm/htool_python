import logging

import Htool
import mpi4py
import numpy as np
from create_geometry import create_random_geometries
from define_custom_dense_blocks_generator import CustomDenseBlocksGenerator
from define_custom_generators import CustomGenerator
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

# Random geometry
size = 500
dimension = 3
[points, _] = create_random_geometries(dimension, size, size)


# Htool parameters
eta = 10
epsilon = 1e-3
minclustersize = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterBuilder()
cluster_builder.set_minclustersize(minclustersize)
target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    points, number_of_children, mpi4py.MPI.COMM_WORLD.size
)


# Build generator
generator = CustomGenerator(points, points)

#
custom_dense_blocks_generator = CustomDenseBlocksGenerator(
    generator, target_cluster, target_cluster
)

# Build HMatrix
# low_rank_generator = CustomSVD(generator)
hmatrix_builder = Htool.HMatrixBuilder(epsilon, eta, "N", "N")

hmatrix_builder.set_dense_blocks_generator(custom_dense_blocks_generator)

# Build distributed operator
distributed_operator_from_hmatrix = Htool.DistributedOperatorFromHMatrix(
    generator, target_cluster, target_cluster, hmatrix_builder, mpi4py.MPI.COMM_WORLD
)

distributed_operator = distributed_operator_from_hmatrix.distributed_operator
hmatrix = distributed_operator_from_hmatrix.hmatrix
Htool.openmp_recompression(hmatrix)

# Test matrix vector product
np.random.seed(0)
x = np.random.rand(size)
y_1 = distributed_operator * x
y_2 = generator.mat_vec(x)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2))


# Test matrix matrix product
X = np.asfortranarray(np.random.rand(size, 2))
Y_1 = distributed_operator @ X
Y_2 = generator.mat_mat(X)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(Y_1 - Y_2) / np.linalg.norm(Y_2))
# Several ways to display information
hmatrix_distributed_information = hmatrix.get_distributed_information(
    mpi4py.MPI.COMM_WORLD
)
hmatrix_tree_parameter = hmatrix.get_tree_parameters()
hmatrix_local_information = hmatrix.get_local_information()
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    print(hmatrix_distributed_information)
    print(hmatrix_local_information)
    print(hmatrix_tree_parameter)

    fig = plt.figure()

    if dimension == 2:
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        # ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    elif dimension == 3:
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        # ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_title("target cluster at depth 1")
    ax2.set_title("target cluster at depth 2")
    # ax3.set_title("source cluster at depth 1")
    ax4.set_title("Hmatrix on rank 0")
    Htool.plot(ax1, target_cluster, points, 1)
    Htool.plot(ax2, target_cluster, points, 2)
    Htool.plot(ax4, hmatrix)
    plt.show()
