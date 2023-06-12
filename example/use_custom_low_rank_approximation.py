import Htool
import mpi4py
import numpy as np
from create_geometry import create_partitionned_geometries
from define_custom_generators import CustomGenerator
from define_custom_low_rank_generator import CustomSVD

# Random geometry
nb_rows = 500
nb_cols = 500
dimension = 3
[target_points, source_points, target_partition] = create_partitionned_geometries(
    dimension, nb_rows, nb_cols, mpi4py.MPI.COMM_WORLD.size
)


# Htool parameters
eta = 100
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

# Low rank generator

low_rank_generator = CustomSVD(generator)

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
    mpi4py.MPI.COMM_WORLD.rank,
)
hmatrix_builder.set_low_rank_generator(low_rank_generator)

# Build distributed operator
distributed_operator_from_hmatrix = Htool.DistributedOperatorFromHMatrix(
    generator, target_cluster, source_cluster, hmatrix_builder, mpi4py.MPI.COMM_WORLD
)

distributed_operator = distributed_operator_from_hmatrix.distributed_operator


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
