import Htool
import matplotlib.pyplot as plt
from create_geometry import create_random_geometries

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, _] = create_random_geometries(dimension, nb_rows, nb_cols)
# [target_points, _, _] = create_partitionned_geometries_test(3, nb_rows, nb_cols, 1)


# Parameters
maximal_leaf_size = 10
number_of_children = 2

# Cluster builder
cluster_builder = Htool.ClusterTreeBuilder()
cluster_builder.set_maximal_leaf_size(maximal_leaf_size)

# Strategies
partitioning_strategy = Htool.PCARegular()
cluster_builder.set_partitioning_strategy(partitioning_strategy)


# Build cluster
target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    target_points, number_of_children
)

fig = plt.figure()

if dimension == 2:
    ax1 = fig.add_subplot(1, 1, 1)
elif dimension == 3:
    ax1 = fig.add_subplot(1, 1, 1, projection="3d")

ax1.set_title("target cluster\ndepth 2")
Htool.plot(ax1, target_cluster, target_points, 2)
plt.show()
