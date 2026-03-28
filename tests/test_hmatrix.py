import copy
import logging

import numpy as np
import pytest

import Htool
from example.advanced.define_custom_low_rank_generator import CustomSVD
from example.create_geometry import create_random_geometries
from example.define_generators import CustomGenerator


@pytest.mark.parametrize(
    "loglevel,symmetry",
    [
        (logging.INFO, "N"),
        (logging.DEBUG, "N"),
        (logging.WARNING, "N"),
        (logging.ERROR, "N"),
        (logging.CRITICAL, "N"),
        (logging.INFO, "S"),
    ],
)
def test_hmatrix(loglevel, symmetry):
    logging.basicConfig(level=loglevel)

    # Random geometry
    nb_rows = 500
    nb_cols = 500
    dimension = 3
    [target_points, source_points] = create_random_geometries(
        dimension, nb_rows, nb_cols
    )
    # Htool parameters
    eta = 100
    epsilon = 1e-3
    maximal_leaf_size = 10
    number_of_children = 2

    # Build clusters
    cluster_builder = Htool.ClusterTreeBuilder()
    cluster_builder.set_maximal_leaf_size(maximal_leaf_size)
    target_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
        target_points,
        number_of_children,
    )
    if symmetry == "N":
        source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
            source_points, number_of_children
        )
    else:
        source_cluster = target_cluster

    # Build generator
    if symmetry == "N":
        generator = CustomGenerator(target_points, source_points)
    else:
        generator = CustomGenerator(target_points, target_points)

    # Custom low rank generator
    low_rank_generator = CustomSVD(generator, False)

    # Build HMatrix
    hmatrix_builder = Htool.HMatrixTreeBuilder(epsilon, eta, "N", "N")
    hmatrix_builder.set_low_rank_generator(low_rank_generator)
    hmatrix = hmatrix_builder.build(generator, target_cluster, source_cluster)
    assert hmatrix.shape == (nb_rows, nb_cols)

    # Copy
    copy_hmatrix = copy.deepcopy(hmatrix)

    # Densifying
    _ = hmatrix.to_dense()
    dense_in_user_numbering = hmatrix.to_dense_in_user_numbering()

    # HMatrix vector product
    np.random.seed(0)
    x = np.random.rand(nb_cols)
    y = hmatrix * x
    y_exact = generator.mat_vec(x)
    y_dense = dense_in_user_numbering.dot(x)
    y_copy = copy_hmatrix * x
    assert np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact) < epsilon
    assert np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense) < 1e-10
    assert np.linalg.norm(y - y_copy) < 1e-10

    # HMatrix matrix product
    np.random.seed(0)
    x = np.random.rand(nb_cols, 2)
    y = hmatrix @ x
    y_exact = generator.mat_mat(x)
    y_dense = dense_in_user_numbering @ x
    y_copy = copy_hmatrix @ x
    assert np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact) < epsilon
    assert np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense) < 1e-10
    assert np.linalg.norm(y - y_copy) < 1e-10

    if symmetry != "N":
        # HLU factorization
        copy_hmatrix.lu_factorization()

        # HLU solve vec
        x_ref = np.ones(nb_cols)
        y = hmatrix * x_ref
        x = copy_hmatrix.lu_solve("N", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # HLU solve mat
        x_ref = np.ones((nb_cols, 2))
        y = hmatrix @ x_ref
        x = copy_hmatrix.lu_solve("N", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # Cholesky factorization
        copy_hmatrix = copy.deepcopy(hmatrix)
        copy_hmatrix.cholesky_factorization("L")

        # Cholesky solve vec
        x_ref = np.ones(nb_cols)
        y = hmatrix * x_ref
        x = copy_hmatrix.cholesky_solve("L", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # Cholesky solve mat
        x_ref = np.ones((nb_cols, 2))
        y = hmatrix @ x_ref
        x = copy_hmatrix.cholesky_solve("L", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

    # Output
    hmatrix_tree_parameter = hmatrix.get_tree_parameters()
    hmatrix_local_information = hmatrix.get_local_information()
    print(hmatrix_tree_parameter)
    print(hmatrix_local_information)

    # Clear low rank matrices stored in low_rank_generator when disallowing copy
    low_rank_generator.clear_data()
