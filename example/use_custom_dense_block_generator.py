import Htool
import mpi4py
import numpy as np
from create_geometry import create_random_geometries
from define_custom_dense_blocks_generator import CustomDenseBlocksGenerator
from define_custom_generators import CustomGenerator

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
target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
    points, number_of_children, mpi4py.MPI.COMM_WORLD.size
)


# Build generator
generator = CustomGenerator(points, points)

#
custom_dense_blocks_generator = CustomDenseBlocksGenerator(
    generator, target_cluster, target_cluster
)

# Build HMatrix
hmatrix_builder = Htool.HMatrixBuilder(
    target_cluster,
    target_cluster,
    epsilon,
    eta,
    "N",
    "N",
    -1,
    mpi4py.MPI.COMM_WORLD.rank,
    mpi4py.MPI.COMM_WORLD.rank,
)

hmatrix_builder.set_dense_blocks_generator(custom_dense_blocks_generator)

# Build distributed operator
distributed_operator_from_hmatrix = Htool.DistributedOperatorFromHMatrix(
    generator, target_cluster, target_cluster, hmatrix_builder, mpi4py.MPI.COMM_WORLD
)

distributed_operator = distributed_operator_from_hmatrix.distributed_operator
hmatrix = distributed_operator_from_hmatrix.hmatrix

# Test matrix vector product
np.random.seed(0)
x = np.random.rand(size)
y_1 = distributed_operator * x
y_2 = generator.mat_vec(x)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2))


# Test matrix matrix product
X = np.asfortranarray(np.random.rand(size, 2))
Y_1 = distributed_operator @ X
Y_2 = generator.mat_mat(X)
print(mpi4py.MPI.COMM_WORLD.rank, np.linalg.norm(Y_1 - Y_2) / np.linalg.norm(Y_2))
