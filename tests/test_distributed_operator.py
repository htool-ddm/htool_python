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
    "nb_rows,nb_cols,symmetry,UPLO,use_default_build,low_rank_approximation,dense_blocks_generator,local_operator,is_partition_given",
    [
        (400, 400, "S", "L", True, False, False, False, False),
        (400, 400, "S", "U", True, False, False, False, False),
        (400, 400, "N", "N", True, False, False, False, False),
        (400, 200, "N", "N", True, False, False, False, False),
        (400, 400, "S", "L", False, True, True, True, False),
        (400, 400, "S", "U", False, True, True, True, False),
        (400, 400, "N", "N", False, True, True, True, False),
        (400, 200, "N", "N", False, True, True, True, False),
        (400, 200, "N", "N", True, False, False, False, True),
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
        default_distributed_operator_holder = default_distributed_operator
        distributed_operator = default_distributed_operator_holder.distributed_operator
        local_hmatrix = default_distributed_operator_holder.hmatrix
        print(local_hmatrix.get_local_information())
    else:
        distributed_operator = custom_distributed_operator

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
