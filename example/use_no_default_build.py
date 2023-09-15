import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_partitionned_geometries
from define_custom_generators import CustomGenerator

import Htool

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, source_points, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
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
    target_points, number_of_children, mpi4py.MPI.COMM_WORLD.size, target_partition
)
source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    source_points, number_of_children, mpi4py.MPI.COMM_WORLD.size
)


# Build generator
generator = CustomGenerator(
    target_cluster, target_points, source_cluster, source_points
)

# Build HMatrix
hmatrix_builder = Htool.HMatrixBuilder(
    target_cluster,
    source_cluster,
    epsilon,
    eta,
    "N",
    "N",
    -1,
    mpi4py.MPI.COMM_WORLD.rank,
)

hmatrix: Htool.HMatrix = hmatrix_builder.build(generator)


# Build local operator
local_operator = Htool.LocalHMatrix(
    hmatrix,
    target_cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank),
    source_cluster,
    "N",
    "N",
    False,
    False,
)

# Build distributed operator
target_partition_from_cluster = Htool.PartitionFromCluster(target_cluster)
source_partition_from_cluster = Htool.PartitionFromCluster(source_cluster)
distributed_operator = Htool.DistributedOperator(
    target_partition_from_cluster,
    source_partition_from_cluster,
    "N",
    "N",
    mpi4py.MPI.COMM_WORLD,
)

distributed_operator.add_local_operator(local_operator)


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


# Outputs
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
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

    ax1.set_title("target cluster at depth 1")
    ax2.set_title("target cluster at depth 2")
    ax3.set_title("source cluster at depth 1")
    ax4.set_title("Hmatrix on rank 0")
    Htool.plot(ax1, target_cluster, target_points, 1)
    Htool.plot(ax2, target_cluster, target_points, 2)
    Htool.plot(ax3, source_cluster, source_points, 1)
    Htool.plot(ax4, hmatrix)
    plt.show()