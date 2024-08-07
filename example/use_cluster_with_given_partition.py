import Htool
import matplotlib.pyplot as plt
import mpi4py
from create_geometry import create_partitionned_geometries

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, _, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
)


# Parameters
minclustersize = 10
number_of_children = 2

# Build clusters
cluster_builder = Htool.ClusterBuilder()
cluster_builder.set_minclustersize(minclustersize)

target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    target_points, number_of_children, mpi4py.MPI.COMM_WORLD.size, target_partition
)

# Local cluster
local_target_cluster: Htool.Cluster = target_cluster.get_cluster_on_partition(
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
    Htool.plot(ax1, target_cluster, target_points, 1)
    Htool.plot(ax2, local_target_cluster, target_points, 1)
    plt.show()
