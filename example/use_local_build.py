import logging

import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_partitionned_geometries, create_random_geometries
from define_custom_generators import CustomGenerator
from define_custom_local_operator import CustomLocalOperator

import Htool

# Random geometry
target_size = 500
source_size = 500
dimension = 3
# [points, _, partition] = create_partitionned_geometries(
#     dimension, size, size, mpi4py.MPI.COMM_WORLD.size
# )

[target_points, source_points] = create_random_geometries(
    dimension, target_size, source_size
)

# Htool parameters
eta = 10
epsilon = 1e-3
minclustersize = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterBuilder()
cluster_builder.set_minclustersize(minclustersize)
target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    target_points,
    number_of_children,
    mpi4py.MPI.COMM_WORLD.size,
)

source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    source_points,
    number_of_children,
    mpi4py.MPI.COMM_WORLD.size,
)

local_target_cluster: Htool.Cluster = target_cluster.get_cluster_on_partition(
    mpi4py.MPI.COMM_WORLD.rank
)
local_source_cluster: Htool.Cluster = source_cluster.get_cluster_on_partition(
    mpi4py.MPI.COMM_WORLD.rank
)

#
source_permutation = source_cluster.get_permutation()
permuted_source_points = np.zeros((dimension, source_size))
for i in range(0, source_size):
    permuted_source_points[:, i] = source_points[:, source_permutation[i]]

# Build generator
generator = CustomGenerator(
    target_cluster, target_points, source_cluster, source_points
)


# Build distributed operator
default_local_approximation = Htool.DefaultLocalApproximationBuilder(
    generator,
    target_cluster,
    source_cluster,
    epsilon,
    eta,
    "N",
    "N",
    mpi4py.MPI.COMM_WORLD,
)
distributed_operator = default_local_approximation.distributed_operator


# Build off diagonal operators
off_diagonal_nc_1 = local_source_cluster.get_offset()
off_diagonal_nc_2 = (
    source_cluster.get_size()
    - local_source_cluster.get_size()
    - local_source_cluster.get_offset()
)
local_nc = local_source_cluster.get_size()

off_diagonal_partition = np.zeros((2, 2), dtype=int)
off_diagonal_partition[0, 0] = 0
off_diagonal_partition[1, 0] = off_diagonal_nc_1
off_diagonal_partition[0, 1] = off_diagonal_nc_1 + local_nc
off_diagonal_partition[1, 1] = off_diagonal_nc_2
off_diagonal_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    permuted_source_points, number_of_children, 2, off_diagonal_partition
)

off_diagonal_generator = CustomGenerator(
    target_cluster,
    target_points,
    off_diagonal_cluster,
    permuted_source_points,
)

local_operator_1 = None
if off_diagonal_nc_1 > 0:
    local_operator_1 = CustomLocalOperator(
        off_diagonal_generator,
        local_target_cluster,
        off_diagonal_cluster.get_cluster_on_partition(0),
        "N",
        "N",
        False,
        True,
    )

local_operator_2 = None
if off_diagonal_nc_2 > 0:
    local_operator_2 = CustomLocalOperator(
        off_diagonal_generator,
        local_target_cluster,
        off_diagonal_cluster.get_cluster_on_partition(1),
        "N",
        "N",
        False,
        True,
    )

if off_diagonal_nc_1 > 0:
    distributed_operator.add_local_operator(local_operator_1)
if off_diagonal_nc_2 > 0:
    distributed_operator.add_local_operator(local_operator_2)

# Test matrix vector product
np.random.seed(0)
x = np.random.rand(source_size)
x = np.ones(source_size)
y_1 = distributed_operator * x
y_2 = generator.mat_vec(x)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2))


# Test matrix matrix product
X = np.asfortranarray(np.random.rand(source_size, 2))
Y_1 = distributed_operator @ X
Y_2 = generator.mat_mat(X)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(Y_1 - Y_2) / np.linalg.norm(Y_2))

# Several ways to display information
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    hmatrix = default_local_approximation.hmatrix
    print(hmatrix.get_tree_parameters())
    print(hmatrix.get_information())

    fig = plt.figure()
    ax1 = None
    ax2 = None
    ax3 = None
    ax4 = None
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

    ax1.set_title("source cluster at depth 1")
    ax2.set_title("source cluster at depth 2")
    ax3.set_title("off diagonal cluster on rank 0 at depth 2")
    ax4.set_title("Hmatrix on rank 0")
    Htool.plot(ax1, source_cluster, source_points, 1)
    Htool.plot(ax2, source_cluster, source_points, 2)
    Htool.plot(
        ax3,
        off_diagonal_cluster.get_cluster_on_partition(1),
        permuted_source_points,
        2,
    )
    Htool.plot(ax4, hmatrix)
    plt.show()
