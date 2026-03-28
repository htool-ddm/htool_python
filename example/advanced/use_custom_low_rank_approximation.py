import logging
import os
import sys

import mpi4py
import numpy as np
from define_custom_low_rank_generator import CustomSVD
from matplotlib import pyplot as plt

import Htool

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_geometry import create_partitionned_geometries
from define_generators import CustomGenerator

logging.basicConfig(level=logging.INFO)

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, source_points, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
)


# Htool parameters
eta = 100
epsilon = 1e-3
maximal_leaf_size = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterTreeBuilder()
cluster_builder.set_maximal_leaf_size(maximal_leaf_size)
target_cluster: Htool.Cluster = (
    cluster_builder.create_cluster_tree_from_local_partition(
        target_points,
        number_of_children,
        mpi4py.MPI.COMM_WORLD.size,
        target_partition,
    )
)
source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    source_points, number_of_children, size_of_partition=mpi4py.MPI.COMM_WORLD.size
)


# Build generator
generator = CustomGenerator(target_points, source_points)

# Low rank generator

low_rank_generator = CustomSVD(generator)

# Build HMatrix
hmatrix_builder = Htool.HMatrixTreeBuilder(epsilon, eta, "N", "N")
# or hmatrix_builder.set_low_rank_generator(low_rank_generator)

# Build distributed operator
distributed_operator_from_hmatrix = Htool.DefaultApproximationBuilder(
    generator, target_cluster, source_cluster, hmatrix_builder, mpi4py.MPI.COMM_WORLD
)

distributed_operator = distributed_operator_from_hmatrix.distributed_operator
hmatrix = distributed_operator_from_hmatrix.hmatrix
Htool.openmp_recompression(hmatrix)

# Test matrix vector product
np.random.seed(0)
x = np.random.rand(nb_cols)
y_1 = distributed_operator * x
y_2 = generator.mat_vec(x)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2))


# Test matrix matrix product
X = np.asfortranarray(np.random.rand(nb_cols, 2))
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
    Htool.plot(ax1, target_cluster, target_points, 1)
    Htool.plot(ax2, target_cluster, target_points, 2)
    Htool.plot(ax3, source_cluster, source_points, 1)
    Htool.plot(ax4, hmatrix)
    plt.show()
