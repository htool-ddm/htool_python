import Htool
import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_partitionned_geometries
from define_custom_generators import CustomGenerator

# Random geometry
size = 500
dimension = 3
[points, _, partition] = create_partitionned_geometries(
    dimension, size, size, mpi4py.MPI.COMM_WORLD.size
)


# Htool parameters
eta = 10
epsilon = 1e-3
minclustersize = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterBuilder()
cluster_builder.set_minclustersize(minclustersize)
cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    points, number_of_children, mpi4py.MPI.COMM_WORLD.size, partition
)


# Build generator
generator = CustomGenerator(cluster, points, cluster, points)

# Build HMatrix
hmatrix_builder = Htool.HMatrixBuilder(
    cluster,
    cluster,
    epsilon,
    eta,
    "S",
    "L",
    -1,
    mpi4py.MPI.COMM_WORLD.rank,
)

hmatrix: Htool.HMatrix = hmatrix_builder.build(generator)


# Build local operator
local_operator = Htool.LocalHMatrix(
    hmatrix,
    cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank),
    cluster,
    "N",
    "N",
    False,
    False,
)

# Build distributed operator
partition_from_cluster = Htool.PartitionFromCluster(cluster)
distributed_operator = Htool.DistributedOperator(
    partition_from_cluster,
    partition_from_cluster,
    "S",
    "L",
    mpi4py.MPI.COMM_WORLD,
)

distributed_operator.add_local_operator(local_operator)


# Solver with block Jacobi preconditionner
block_diagonal_hmatrix = hmatrix.get_sub_hmatrix(
    cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank),
    cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank),
)
default_solver_builder = Htool.DefaultSolverBuilder(
    distributed_operator,
    block_diagonal_hmatrix,
)
solver = default_solver_builder.solver

# Solver with block Jacobi
x_ref = np.random.random(size)
b = distributed_operator * x_ref
x = np.zeros(size)

hpddm_args = "-hpddm_compute_residual l2 "
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    hpddm_args += "-hpddm_verbosity 10"
solver.set_hpddm_args(hpddm_args)
solver.facto_one_level()
solver.solve(x, b)


# Outputs
hmatrix_distributed_information = hmatrix.get_distributed_information(
    mpi4py.MPI.COMM_WORLD
)
hmatrix_tree_parameter = hmatrix.get_tree_parameters()
hmatrix_local_information = hmatrix.get_local_information()
solver_information = solver.get_information()
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    print(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    print(hmatrix_distributed_information)
    print(hmatrix_local_information)
    print(hmatrix_tree_parameter)
    print(solver_information)

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
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

    ax1.set_title("cluster at depth 1")
    ax2.set_title("cluster at depth 2")
    ax3.set_title("Hmatrix on rank 0")
    ax4.set_title("Block diagonal Hmatrix on rank 0")
    Htool.plot(ax1, cluster, points, 1)
    Htool.plot(ax2, cluster, points, 2)
    Htool.plot(ax3, hmatrix)
    Htool.plot(ax4, block_diagonal_hmatrix)
    plt.show()
