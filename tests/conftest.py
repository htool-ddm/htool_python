import os
import pathlib
import struct

import mpi4py
import numpy as np
import pytest

import Htool
from example.advanced.define_custom_dense_blocks_generator import (
    CustomDenseBlocksGenerator,
)
from example.advanced.define_custom_local_operator import (
    CustomLocalToLocalOperator,
    CustomRestrictedGlobalToLocalOperator,
)
from example.advanced.define_custom_low_rank_generator import CustomSVD
from example.define_generators import CustomGenerator


class GeneratorFromMatrix(Htool.VirtualGenerator):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def get_coef(self, i, j):
        return self.matrix[i, j]

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = self.get_coef(J[j], K[k])


class ComplexGeneratorFromMatrix(Htool.ComplexVirtualGenerator):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = matrix

    def get_coef(self, i, j):
        return self.matrix[i, j]

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = self.get_coef(J[j], K[k])


class LocalGeneratorFromMatrix(Htool.VirtualGenerator):
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


class ComplexLocalGeneratorFromMatrix(Htool.ComplexVirtualGenerator):
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
def geometry(partition_type: str, dimension: int, nb_rows: int, nb_cols: int, symmetry):
    np.random.seed(0)
    sizeWorld = mpi4py.MPI.COMM_WORLD.size
    target_points = None
    source_points = None
    target_partition = None

    if partition_type != "None":
        target_points = np.zeros((dimension, nb_rows))
        target_local_size = int(nb_rows / sizeWorld)
        target_partition = np.zeros(
            (2, sizeWorld),
        ).astype(int)

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
def cluster(geometry, symmetry, partition_type, number_of_children):
    # Parameters
    [target_points, source_points, target_partition] = geometry
    maximal_leaf_size = 10

    # Build clusters
    cluster_builder = Htool.ClusterTreeBuilder()
    cluster_builder.set_maximal_leaf_size(maximal_leaf_size)
    source_cluster = None
    if symmetry == "N":
        source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            source_points,
            number_of_children,
            size_of_partition=mpi4py.MPI.COMM_WORLD.size,
            radii=None,
            weights=None,
        )

    if target_partition is not None:
        if partition_type == "Local":
            target_cluster: Htool.Cluster = (
                cluster_builder.create_cluster_tree_from_local_partition(
                    target_points,
                    number_of_children,
                    mpi4py.MPI.COMM_WORLD.size,
                    target_partition,
                    radii=None,
                    weights=None,
                )
            )
        else:
            global_partition = np.zeros(target_points.shape[1])
            for i in range(0, mpi4py.MPI.COMM_WORLD.size):
                global_partition[
                    target_partition[0, i] : target_partition[0, i]
                    + target_partition[1, i]
                ] = i
            target_cluster: Htool.Cluster = (
                cluster_builder.create_cluster_tree_from_global_partition(
                    target_points,
                    number_of_children,
                    mpi4py.MPI.COMM_WORLD.size,
                    global_partition,
                    radii=None,
                    weights=None,
                )
            )
    else:
        target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            target_points,
            number_of_children,
            size_of_partition=mpi4py.MPI.COMM_WORLD.size,
            radii=None,
            weights=None,
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


@pytest.fixture(params=["None", "ExtraDiagonal", "LocalAndExtraDiagonal"])
def local_operator(request, generator, cluster, geometry):
    if request.param == "ExtraDiagonal" or request.param == "LocalAndExtraDiagonal":
        [target_points, source_points, _] = geometry
        [target_cluster, source_cluster] = cluster
        result = ["ExtraDiagonal", []]
        target_local_cluster = target_cluster.get_cluster_on_partition(
            mpi4py.MPI.COMM_WORLD.rank
        )
        source_local_cluster = source_cluster.get_cluster_on_partition(
            mpi4py.MPI.COMM_WORLD.rank
        )
        if source_local_cluster.get_offset() > 0:
            result[1].append(
                CustomRestrictedGlobalToLocalOperator(
                    generator,
                    Htool.LocalRenumbering(target_local_cluster),
                    Htool.LocalRenumbering(
                        0,
                        source_local_cluster.get_offset(),
                        source_cluster.get_permutation(),
                    ),
                    False,
                    False,
                )
            )
        if (
            source_cluster.get_size()
            - source_local_cluster.get_size()
            - source_local_cluster.get_offset()
            > 0
        ):
            result[1].append(
                CustomRestrictedGlobalToLocalOperator(
                    generator,
                    Htool.LocalRenumbering(target_local_cluster),
                    Htool.LocalRenumbering(
                        source_local_cluster.get_size()
                        + source_local_cluster.get_offset(),
                        source_cluster.get_size()
                        - source_local_cluster.get_size()
                        - source_local_cluster.get_offset(),
                        source_cluster.get_permutation(),
                    ),
                    False,
                    False,
                )
            )
            assert (
                result[1][-1].local_target_renumbering.size
                == target_local_cluster.get_size()
            )
            assert (
                result[1][-1].local_source_renumbering.size
                == source_cluster.get_size()
                - source_local_cluster.get_size()
                - source_local_cluster.get_offset()
            )

        if request.param == "LocalAndExtraDiagonal":
            result[0] = "LocalAndExtraDiagonal"
            result[1].append(
                CustomLocalToLocalOperator(
                    generator,
                    Htool.LocalRenumbering(target_local_cluster),
                    Htool.LocalRenumbering(source_local_cluster),
                )
            )
        return result
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

    return [
        target_cluster,
        source_cluster,
        Htool.DefaultApproximationBuilder(
            generator,
            target_cluster,
            source_cluster,
            Htool.HMatrixTreeBuilder(epsilon, eta, symmetry, UPLO),
            mpi4py.MPI.COMM_WORLD,
        ),
    ]


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
    if local_operator is None:
        hmatrix_tree_builder = Htool.HMatrixTreeBuilder(
            epsilon,
            eta,
            symmetry,
            UPLO,
        )
        if dense_blocks_generator is not None:
            hmatrix_tree_builder.set_dense_blocks_generator(dense_blocks_generator)
        if low_rank_approximation is not None:
            hmatrix_tree_builder.set_low_rank_generator(low_rank_approximation)

        distributed_operator_holder = Htool.DefaultApproximationBuilder(
            generator,
            target_cluster,
            source_cluster,
            hmatrix_tree_builder,
            mpi4py.MPI.COMM_WORLD,
        )

    elif local_operator[0] == "ExtraDiagonal":
        distributed_operator_holder = Htool.DefaultLocalApproximationBuilder(
            generator,
            target_cluster,
            source_cluster,
            Htool.HMatrixTreeBuilder(epsilon, eta, symmetry, UPLO),
            mpi4py.MPI.COMM_WORLD,
        )
        for op in local_operator[1]:
            distributed_operator_holder.distributed_operator.add_global_to_local_operator(
                op
            )
    elif local_operator[0] == "LocalAndExtraDiagonal":
        distributed_operator_holder = Htool.CustomApproximationBuilder(
            target_cluster, source_cluster, mpi4py.MPI.COMM_WORLD, local_operator[1][-1]
        )
        for op in local_operator[1][0:-1]:
            distributed_operator_holder.distributed_operator.add_global_to_local_operator(
                op
            )

    return [target_cluster, source_cluster, distributed_operator_holder]


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

    if symmetry == "S":
        A = A.real

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

    if symmetry == "S":
        f = f.real

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

    if symmetry == "S":
        x_ref = x_ref.real

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
            local_neumann_matrix = local_neumann_matrix.real
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
