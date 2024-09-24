import logging

import Htool
import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_random_geometries
from define_custom_local_operator import CustomGlobalToLocalOperator
from define_generators import CustomGenerator

logging.basicConfig(level=logging.INFO)

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
maximal_leaf_size = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterTreeBuilder()
cluster_builder.set_maximal_leaf_size(maximal_leaf_size)
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
generator = CustomGenerator(target_points, source_points)


# Build distributed operator
default_local_approximation = Htool.DefaultLocalApproximationBuilder(
    generator,
    target_cluster,
    source_cluster,
    Htool.HMatrixTreeBuilder(epsilon, eta, "N", "N"),
    mpi4py.MPI.COMM_WORLD,
)
distributed_operator = default_local_approximation.distributed_operator
hmatrix = default_local_approximation.hmatrix
Htool.recompression(hmatrix)

local_operator_1 = None
if local_source_cluster.get_offset() > 0:
    local_operator_1 = CustomGlobalToLocalOperator(
        generator,
        Htool.LocalRenumbering(local_target_cluster),
        Htool.LocalRenumbering(
            0, local_source_cluster.get_offset(), source_cluster.get_permutation()
        ),
    )

local_operator_2 = None
if (
    source_cluster.get_size()
    - local_source_cluster.get_size()
    - local_source_cluster.get_offset()
    > 0
):
    local_operator_2 = CustomGlobalToLocalOperator(
        generator,
        Htool.LocalRenumbering(local_target_cluster),
        Htool.LocalRenumbering(
            local_source_cluster.get_size() + local_source_cluster.get_offset(),
            source_cluster.get_size()
            - local_source_cluster.get_size()
            - local_source_cluster.get_offset(),
            source_cluster.get_permutation(),
        ),
    )

if local_operator_1:
    distributed_operator.add_global_to_local_operator(local_operator_1)
if local_operator_2:
    distributed_operator.add_global_to_local_operator(local_operator_2)

# Test matrix vector product
np.random.seed(0)
x = np.random.rand(source_size)
y_1 = distributed_operator * x
y_2 = generator.mat_vec(x)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2))


# Test matrix matrix product
X = np.asfortranarray(np.random.rand(source_size, 5))
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
        ax3 = fig.add_subplot(2, 2, 4)
    elif dimension == 3:
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax3 = fig.add_subplot(2, 2, 4)

    ax1.set_title("source cluster at depth 1")
    ax2.set_title("source cluster at depth 2")
    ax3.set_title("Hmatrix on rank 0")
    Htool.plot(ax1, source_cluster, source_points, 1)
    Htool.plot(ax2, source_cluster, source_points, 2)
    Htool.plot(ax3, hmatrix)
    plt.show()
