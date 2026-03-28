import matplotlib.pyplot as plt
import mpi4py
import numpy as np
import pytest

import Htool


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("nb_rhs", [1, 5])
# @pytest.mark.parametrize(
#     "use_default_build",
#     [True, False],
#     ids=["default_hmatrix_build", "custom_hmatrix_build"],
# )
@pytest.mark.parametrize(
    "nb_rows,nb_cols,symmetry,UPLO,use_default_build,low_rank_approximation,dense_blocks_generator,local_operator,partition_type,number_of_children",
    [
        (400, 400, "S", "L", True, False, False, "None", "None", 2),
        (400, 400, "S", "U", True, False, False, "None", "None", 2),
        (400, 400, "N", "N", True, False, False, "None", "None", 2),
        (400, 200, "N", "N", True, False, False, "None", "None", 2),
        (400, 400, "S", "L", False, True, True, "None", "None", 2),
        (400, 400, "S", "U", False, True, True, "None", "None", 2),
        (400, 400, "N", "N", False, True, True, "None", "None", 2),
        (400, 200, "N", "N", False, True, True, "None", "None", 2),
        (400, 400, "S", "L", False, False, False, "ExtraDiagonal", "None", 2),
        (400, 400, "S", "U", False, False, False, "ExtraDiagonal", "None", 2),
        (400, 400, "N", "N", False, False, False, "ExtraDiagonal", "None", 2),
        (400, 200, "N", "N", False, False, False, "ExtraDiagonal", "None", 2),
        (400, 400, "S", "L", False, False, False, "LocalAndExtraDiagonal", "None", 2),
        (400, 400, "S", "U", False, False, False, "LocalAndExtraDiagonal", "None", 2),
        (400, 400, "N", "N", False, False, False, "LocalAndExtraDiagonal", "None", 2),
        (400, 200, "N", "N", False, False, False, "LocalAndExtraDiagonal", "None", 2),
        (400, 200, "N", "N", True, False, False, "None", "Local", 2),
    ],
    indirect=["low_rank_approximation", "dense_blocks_generator", "local_operator"],
)
def test_distributed_operator(
    nb_cols,
    nb_rhs,
    epsilon,
    generator,
    use_default_build,
    default_distributed_operator,
    custom_distributed_operator,
):
    default_distributed_operator_holder = None
    distributed_operator = None
    if use_default_build:
        target_cluster, source_cluster, default_distributed_operator_holder = (
            default_distributed_operator
        )
        distributed_operator = default_distributed_operator_holder.distributed_operator
        local_hmatrix = default_distributed_operator_holder.hmatrix

        hmatrix_distributed_information = local_hmatrix.get_distributed_information(
            mpi4py.MPI.COMM_WORLD
        )
        hmatrix_tree_parameter = local_hmatrix.get_tree_parameters()
        hmatrix_local_information = local_hmatrix.get_local_information()
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            print(hmatrix_distributed_information)
            print(hmatrix_local_information)
            print(hmatrix_tree_parameter)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        Htool.plot(ax1, local_hmatrix)
        plt.close(fig)

        global_target_size = mpi4py.MPI.COMM_WORLD.allreduce(
            local_hmatrix.shape[0], op=mpi4py.MPI.SUM
        )
        assert distributed_operator.shape == (
            global_target_size,
            local_hmatrix.shape[1],
        )
    else:
        target_cluster, source_cluster, distributed_operator_holder = (
            custom_distributed_operator
        )
        distributed_operator = distributed_operator_holder.distributed_operator

    # Test matrix vector product
    np.random.seed(0)
    x = np.random.rand(nb_cols)
    y_1 = distributed_operator * x
    y_2 = generator.mat_vec(x)
    assert np.linalg.norm(y_1 - y_2) / np.linalg.norm(y_2) < epsilon

    # Test matrix matrix product
    X = np.asfortranarray(np.random.rand(nb_cols, nb_rhs))
    Y_1 = distributed_operator @ X
    Y_2 = generator.mat_mat(X)
    assert np.linalg.norm(Y_1 - Y_2) / np.linalg.norm(Y_2) < epsilon

    X = np.asfortranarray(np.random.rand(nb_cols, 1))
    Y_1 = distributed_operator @ X
    Y_2 = generator.mat_mat(X)
    assert np.linalg.norm(Y_1 - Y_2) / np.linalg.norm(Y_2) < epsilon

    # Test sub matrix vector product
    test_offset = int(nb_cols / 10)
    test_size = int(nb_cols / 10)
    x[0:test_offset] = 0
    x[test_offset + test_size :] = 0
    x_perm = np.zeros(nb_cols)
    source_permutation = source_cluster.get_permutation()
    x_perm[source_permutation] = x

    y_1 = distributed_operator.internal_sub_vector_product_global_to_local(
        x[test_offset : test_offset + test_size], test_offset
    )
    y_2_perm = generator.mat_vec(x_perm)
    target_permutation = target_cluster.get_permutation()
    y_2 = y_2_perm[target_permutation]
    local_target_cluster = target_cluster.get_cluster_on_partition(
        mpi4py.MPI.COMM_WORLD.Get_rank()
    )
    target_offset = local_target_cluster.get_offset()
    target_size = local_target_cluster.get_size()
    assert (
        np.linalg.norm(y_1 - y_2[target_offset : target_offset + target_size])
        / np.linalg.norm(y_2)
        < (1 + 10) * epsilon
    )
