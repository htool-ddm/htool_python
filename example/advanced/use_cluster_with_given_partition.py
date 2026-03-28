import os
import sys

import matplotlib.pyplot as plt
import mpi4py
import numpy as np

import Htool

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from create_geometry import create_partitionned_geometries

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, _, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
)


# Parameters
maximal_leaf_size = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterTreeBuilder()
cluster_builder.set_maximal_leaf_size(maximal_leaf_size)

cluster: Htool.Cluster = cluster_builder.create_cluster_tree_from_local_partition(
    target_points,
    number_of_children,
    mpi4py.MPI.COMM_WORLD.size,
    target_partition,
)

# Alternatively, use a global definition of partition
global_partition = np.zeros(nb_rows)
for i in range(0, mpi4py.MPI.COMM_WORLD.size):
    global_partition[
        target_partition[0, i] : target_partition[0, i] + target_partition[1, i]
    ] = i
if mpi4py.MPI.COMM_WORLD.rank == 0:
    print(global_partition)
cluster_2: Htool.Cluster = cluster_builder.create_cluster_tree_from_global_partition(
    target_points,
    number_of_children,
    mpi4py.MPI.COMM_WORLD.size,
    global_partition,
)

# Local cluster
local_cluster: Htool.Cluster = cluster.get_cluster_on_partition(
    mpi4py.MPI.COMM_WORLD.rank
)

if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    fig = plt.figure()

    if dimension == 2:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    elif dimension == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.set_title("target cluster\ndepth 2")
    ax2.set_title("local cluster\ntarget partition number 0\ndepth 1")
    Htool.plot(ax1, cluster, target_points, 1)
    Htool.plot(ax2, local_cluster, target_points, 1)
    plt.show()
