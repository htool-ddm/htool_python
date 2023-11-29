import matplotlib.pyplot as plt
import mpi4py
import numpy as np
from create_geometry import create_random_geometries

import Htool

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, _] = create_random_geometries(dimension, nb_rows, nb_cols)
# [target_points, _, _] = create_partitionned_geometries_test(3, nb_rows, nb_cols, 1)


# Parameters
minclustersize = 10
number_of_children = 2

# Cluster builder
cluster_builder = Htool.ClusterBuilder()
cluster_builder.set_minclustersize(minclustersize)

# Strategies
# direction_computation_strategy = Htool.ComputeBoundingBox()
direction_computation_strategy = Htool.ComputeLargestExtent()  # default
splitting_strategy = Htool.RegularSplitting()  # default
# splitting_strategy = Htool.GeometricSplitting()
cluster_builder.set_direction_computation_strategy(direction_computation_strategy)
cluster_builder.set_splitting_strategy(splitting_strategy)


# Build cluster
target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    target_points, number_of_children, mpi4py.MPI.COMM_WORLD.size
)


# Local cluster
local_target_cluster: Htool.Cluster = target_cluster.get_cluster_on_partition(
    mpi4py.MPI.COMM_WORLD.rank
)

if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    fig = plt.figure()
    ax1 = None
    ax2 = None

    if dimension == 2:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
    elif dimension == 3:
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.set_title("target cluster\ndepth 2")
    ax2.set_title("local cluster\ntarget partition number 0\ndepth 2")
    Htool.plot(ax1, target_cluster, target_points, 2)
    Htool.plot(ax2, local_target_cluster, target_points, 2)
    plt.show()
