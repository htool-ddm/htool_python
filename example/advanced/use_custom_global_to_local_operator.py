import os
import sys

import mpi4py
import numpy as np
from define_custom_local_operator import CustomRestrictedGlobalToLocalOperator

import Htool

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_geometry import create_partitionned_geometries
from define_generators import CustomGenerator

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, source_points, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
)


# Htool parameters
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

# Build local operator
local_cluster = target_cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank)
local_operator = CustomRestrictedGlobalToLocalOperator(
    generator,
    Htool.LocalRenumbering(local_cluster),
    Htool.LocalRenumbering(source_cluster),
)

# Build distributed operator
custom_local_approximation = Htool.CustomApproximationBuilder(
    target_cluster, source_cluster, mpi4py.MPI.COMM_WORLD, local_operator
)
distributed_operator = custom_local_approximation.distributed_operator

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
