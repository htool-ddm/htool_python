import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_random_geometries
from define_custom_generators import CustomGenerator

import Htool

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
cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    points, number_of_children, mpi4py.MPI.COMM_WORLD.size
)

# Build generator
generator = CustomGenerator(cluster, points, cluster, points)

# Build distributed operator
default_approximation = Htool.DefaultApproximationBuilder(
    generator,
    cluster,
    cluster,
    epsilon,
    eta,
    "S",
    "L",
    mpi4py.MPI.COMM_WORLD,
)

# Solver with block Jacobi preconditionner
default_solver_builder = Htool.DefaultSolverBuilder(
    default_approximation.distributed_operator,
    default_approximation.block_diagonal_hmatrix,
)
solver = default_solver_builder.solver


# Solver with block Jacobi
x_ref = np.random.random(size)
b = default_approximation.distributed_operator * x_ref
x = np.zeros(size)

hpddm_args = "-hpddm_compute_residual l2 "
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    hpddm_args += "-hpddm_verbosity 10"
solver.set_hpddm_args(hpddm_args)
solver.facto_one_level()
solver.solve(x, b)


# Several ways to display information
hmatrix = default_approximation.hmatrix
hmatrix_distributed_information = hmatrix.get_distributed_information(
    mpi4py.MPI.COMM_WORLD
)
hmatrix_tree_parameter = hmatrix.get_tree_parameters()
hmatrix_local_information = hmatrix.get_local_information()
solver_information = solver.get_information()
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    print(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
    hmatrix = default_approximation.hmatrix
    local_block_hmatrix = default_approximation.block_diagonal_hmatrix
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
    ax4.set_title("Hmatrix on rank 0")
    ax4.set_title("Block diagonal Hmatrix on rank 0")
    Htool.plot(ax1, cluster, points, 1)
    Htool.plot(ax2, cluster, points, 2)
    Htool.plot(ax3, hmatrix)
    Htool.plot(ax4, local_block_hmatrix)
    plt.show()
