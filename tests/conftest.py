import os
import pathlib
import struct

import Htool
import mpi4py
import numpy as np
import pytest

from example.define_custom_dense_blocks_generator import CustomDenseBlocksGenerator
from example.define_custom_generators import CustomGenerator
from example.define_custom_local_operator import CustomLocalOperator
from example.define_custom_low_rank_generator import CustomSVD


class GeneratorFromMatrix(Htool.ComplexVirtualGeneratorInUserNumbering):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def get_coef(self, i, j):
        return self.matrix[i, j]

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = self.get_coef(J[j], K[k])


class LocalGeneratorFromMatrix(Htool.ComplexVirtualGeneratorInUserNumbering):
    def __init__(
        self,
        permutation,
        local_to_global_numbering,
        matrix,
    ):
        super().__init__(permutation, permutation)
        self.matrix = matrix
        self.local_to_global_numbering = local_to_global_numbering

    def get_coef(self, i, j):
        return self.matrix[i, j]

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = self.get_coef(
                    self.local_to_global_numbering[J[j]],
                    self.local_to_global_numbering[K[k]],
                )


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if mpi4py.MPI.COMM_WORLD.Get_rank() != 0:
        # unregister the standard reporter for all nonzero ranks
        standard_reporter = config.pluginmanager.getplugin("terminalreporter")
        config.pluginmanager.unregister(standard_reporter)


@pytest.fixture
def geometry(
    is_partition_given: bool, dimension: int, nb_rows: int, nb_cols: int, symmetry
):
    np.random.seed(0)
    sizeWorld = mpi4py.MPI.COMM_WORLD.size
    target_points = None
    source_points = None
    target_partition = None

    if is_partition_given:
        target_points = np.zeros((dimension, nb_rows))
        target_local_size = int(nb_rows / sizeWorld)
        target_partition = np.zeros((2, sizeWorld))

        for i in range(0, sizeWorld - 1):
            target_partition[0, i] = i * target_local_size
            target_partition[1, i] = target_local_size
            target_points[0, i * target_local_size : (i + 1) * target_local_size] = i

        target_points[0, (sizeWorld - 1) * target_local_size :] = sizeWorld - 1
        target_partition[0, sizeWorld - 1] = (sizeWorld - 1) * target_local_size
        target_partition[1, sizeWorld - 1] = (
            nb_rows - (sizeWorld - 1) * target_local_size
        )

        target_points[1:, :] = np.random.rand(dimension - 1, nb_rows)
    else:
        target_points = np.random.random((dimension, nb_rows))

    if symmetry == "N":
        source_points = np.random.random((dimension, nb_cols))
    elif symmetry == "S" or symmetry == "H":
        source_points = target_points

    return [target_points, source_points, target_partition]


@pytest.fixture
def cluster(geometry, symmetry):
    # Parameters
    [target_points, source_points, target_partition] = geometry
    minclustersize = 10
    number_of_children = 2

    # Build clusters
    cluster_builder = Htool.ClusterBuilder()
    cluster_builder.set_minclustersize(minclustersize)
    source_cluster = None
    if symmetry == "N":
        source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            source_points, number_of_children, mpi4py.MPI.COMM_WORLD.size
        )

    if target_partition is not None:
        target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            target_points,
            number_of_children,
            mpi4py.MPI.COMM_WORLD.size,
            target_partition,
        )
    else:
        target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            target_points, number_of_children, mpi4py.MPI.COMM_WORLD.size
        )

    if symmetry == "S" or symmetry == "H":
        source_cluster = target_cluster

    return [target_cluster, source_cluster]


@pytest.fixture
def generator(geometry, cluster):
    [target_points, source_points, _] = geometry
    [target_cluster, source_cluster] = cluster
    return CustomGenerator(target_points, source_points)


@pytest.fixture(
    params=[True, False],
    ids=["custom_dense_block_generator", "auto_dense_block_generator"],
)
def dense_blocks_generator(request, generator, cluster):
    [target_cluster, source_cluster] = cluster
    if request.param:
        return CustomDenseBlocksGenerator(generator, target_cluster, source_cluster)
    else:
        return None


@pytest.fixture(params=[True, False])
def local_operator(request, generator, cluster, geometry):
    if request.param:
        [target_points, source_points, _] = geometry
        [target_cluster, source_cluster] = cluster
        return CustomLocalOperator(
            generator,
            target_cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank),
            source_cluster,
        )
    else:
        return None


@pytest.fixture(
    params=[True, False],
    ids=["custom_low_rank_approximation", "auto_low_rank_approximation"],
)
def low_rank_approximation(request, generator, cluster, epsilon):
    if request.param:
        return CustomSVD(generator)
    else:
        return None


@pytest.fixture
def default_distributed_operator(
    cluster, generator, epsilon: float, eta: float, symmetry, UPLO
):
    [target_cluster, source_cluster] = cluster

    return Htool.DefaultApproximationBuilder(
        generator,
        target_cluster,
        source_cluster,
        epsilon,
        eta,
        symmetry,
        UPLO,
        mpi4py.MPI.COMM_WORLD,
    )


@pytest.fixture
def custom_distributed_operator(
    cluster,
    generator,
    epsilon: float,
    eta: float,
    symmetry,
    UPLO,
    local_operator,
    dense_blocks_generator,
    low_rank_approximation,
):
    [target_cluster, source_cluster] = cluster

    if local_operator is not None:
        distributed_operator_holder = Htool.CustomApproximationBuilder(
            target_cluster,
            source_cluster,
            symmetry,
            UPLO,
            mpi4py.MPI.COMM_WORLD,
            local_operator,
        )
    else:
        hmatrix_builder = Htool.HMatrixBuilder(
            target_cluster,
            source_cluster,
            epsilon,
            eta,
            symmetry,
            UPLO,
            -1,
            mpi4py.MPI.COMM_WORLD.rank,
            mpi4py.MPI.COMM_WORLD.rank,
        )
        if dense_blocks_generator is not None:
            hmatrix_builder.set_dense_blocks_generator(dense_blocks_generator)
        if low_rank_approximation is not None:
            hmatrix_builder.set_low_rank_generator(low_rank_approximation)

        distributed_operator_holder = Htool.DistributedOperatorFromHMatrix(
            generator,
            target_cluster,
            source_cluster,
            hmatrix_builder,
            mpi4py.MPI.COMM_WORLD,
        )

    return distributed_operator_holder


@pytest.fixture()
def load_data_solver(symmetry, mu):
    # MPI
    comm = mpi4py.MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # args
    folder = "non_sym"
    UPLO = "N"
    if symmetry == "S":
        folder = "sym"
        UPLO = "L"

    #
    path_to_data = pathlib.Path(
        os.path.dirname(__file__)
        + "/../lib/htool_generate_data_test/data/output_"
        + folder
    )
    # Matrix
    with open(
        path_to_data / "matrix.bin",
        "rb",
    ) as input:
        data = input.read()
        (m, n) = struct.unpack("@II", data[:8])
        # print(m,n)
        A = np.frombuffer(data[8:], dtype=np.dtype("complex128"))
        A = np.transpose(A.reshape((m, n)))

    # Geometry
    with open(
        path_to_data / "geometry.bin",
        "rb",
    ) as input:
        data = input.read()
        geometry = np.frombuffer(data[4:], dtype=np.dtype("double"))
        geometry = geometry.reshape(3, m, order="F")

    # Right-hand side
    with open(
        path_to_data / "rhs.bin",
        "rb",
    ) as input:
        data = input.read()
        # l = struct.unpack("@I", data[:4])
        rhs = np.frombuffer(data[4:], dtype=np.dtype("complex128"))
    f = np.zeros(len(rhs), dtype="complex128")
    if mu > 1:
        f = np.zeros((len(rhs), mu), dtype="complex128")
        for p in range(0, mu):
            f[:, p] = rhs
    else:
        f = rhs

    # Cluster
    cluster = Htool.read_cluster_from(
        str(path_to_data / ("cluster_" + str(size) + "_cluster_tree_properties.csv")),
        str(path_to_data / ("cluster_" + str(size) + "_cluster_tree.csv")),
    )

    # Global vectors
    with open(
        path_to_data / "sol.bin",
        "rb",
    ) as input:
        data = input.read()
        x_ref = np.frombuffer(data[4:], dtype=np.dtype("complex128"))

    # Domain decomposition
    with open(
        path_to_data
        / ("cluster_to_ovr_subdomain_" + str(size) + "_" + str(rank) + ".bin"),
        "rb",
    ) as input:
        data = input.read()
        cluster_to_ovr_subdomain = np.frombuffer(data[4:], dtype=np.dtype("int32"))
    with open(
        path_to_data
        / ("ovr_subdomain_to_global_" + str(size) + "_" + str(rank) + ".bin"),
        "rb",
    ) as input:
        data = input.read()
        ovr_subdomain_to_global = np.frombuffer(data[4:], dtype=np.dtype("int32"))
    neighbors = []
    with open(
        path_to_data / ("neighbors_" + str(size) + "_" + str(rank) + ".bin"),
        "rb",
    ) as input:
        data = input.read()
        neighbors = np.frombuffer(data[4:], dtype=np.dtype("int32"))

    intersections = []
    for p in range(0, len(neighbors)):
        with open(
            path_to_data
            / ("intersections_" + str(size) + "_" + str(rank) + "_" + str(p) + ".bin"),
            "rb",
        ) as input:
            data = input.read()
            intersection = np.frombuffer(data[4:], dtype=np.dtype("int32"))
            intersections.append(intersection)

    local_neumann_matrix = None
    if symmetry == "S" and size > 1:
        # Matrix
        with open(
            path_to_data / ("Ki_" + str(size) + "_" + str(rank) + ".bin"),
            "rb",
        ) as input:
            data = input.read()
            (m, n) = struct.unpack("@II", data[:8])
            # print(m,n)
            local_neumann_matrix = np.frombuffer(data[8:], dtype=np.dtype("complex128"))
            local_neumann_matrix = np.transpose(
                local_neumann_matrix.reshape((m, n), order="C")
            ).copy("F")
    return [
        A,
        x_ref,
        f,
        geometry,
        cluster,
        neighbors,
        intersections,
        symmetry,
        UPLO,
        cluster_to_ovr_subdomain,
        ovr_subdomain_to_global,
        local_neumann_matrix,
    ]
