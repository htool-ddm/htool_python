import Htool
import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_partitionned_geometries
from define_custom_generators import CustomGenerator

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
generator = CustomGenerator(target_points, source_points)

# Build distributed operator
default_approximation = Htool.DefaultApproximationBuilder(
    generator,
    target_cluster,
    source_cluster,
    epsilon,
    eta,
    "N",
    "N",
    mpi4py.MPI.COMM_WORLD,
)

distributed_operator = default_approximation.distributed_operator

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
hmatrix = default_approximation.hmatrix
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
