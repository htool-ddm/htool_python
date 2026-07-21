import copy
import logging

import numpy as np
import pytest

import Htool
from example.advanced.define_custom_low_rank_generator import (
    ComplexCustomSVD,
    CustomSVD,
)
from example.create_geometry import create_random_geometries
from example.define_generators import ComplexCustomGenerator, CustomGenerator


@pytest.mark.parametrize(
    "loglevel,symmetry,is_complex,policy_name",
    [
        (logging.INFO, "N", False, None),
        (logging.DEBUG, "N", False, None),
        (logging.WARNING, "N", False, None),
        (logging.ERROR, "N", False, None),
        (logging.CRITICAL, "N", False, None),
        (logging.INFO, "S", False, None),
        (logging.INFO, "N", False, "par"),
        (logging.INFO, "N", False, "seq"),
        (logging.INFO, "N", False, "omp_task"),
        (logging.INFO, "S", True, None),
    ],
)
def test_hmatrix(loglevel, symmetry, is_complex, policy_name):
    logging.basicConfig(level=loglevel)

    policy = None
    if policy_name == "par":
        policy = Htool.ParallelPolicy()
    elif policy_name == "seq":
        policy = Htool.SequentialPolicy()
    elif policy_name == "omp_task":
        policy = Htool.ComplexOmpTaskPolicy() if is_complex else Htool.OmpTaskPolicy()

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
    if is_complex is False:
        if symmetry == "N":
            generator = CustomGenerator(target_points, source_points)
        else:
            generator = CustomGenerator(target_points, target_points)
    else:
        if symmetry == "N":
            generator = ComplexCustomGenerator(target_points, source_points)
        else:
            generator = ComplexCustomGenerator(target_points, target_points)
    # Custom low rank generator
    if is_complex is False:
        low_rank_generator = CustomSVD(generator, False)
    else:
        low_rank_generator = ComplexCustomSVD(generator, False)

    # Build HMatrix
    if is_complex is False:
        hmatrix_builder = Htool.HMatrixTreeBuilder(epsilon, eta, "N", "N")
    else:
        hmatrix_builder = Htool.ComplexHMatrixTreeBuilder(epsilon, eta, "N", "N")
    hmatrix_builder.set_low_rank_generator(low_rank_generator)
    if symmetry == "S":
        hmatrix_builder.set_block_tree_consistency(True)

    if policy:
        hmatrix = hmatrix_builder.build(
            policy, generator, target_cluster, source_cluster
        )
    else:
        hmatrix = hmatrix_builder.build(generator, target_cluster, source_cluster)
    assert hmatrix.shape == (nb_rows, nb_cols)

    # Copy
    copy_hmatrix = copy.deepcopy(hmatrix)

    # Densifying
    _ = hmatrix.to_dense()
    dense_in_user_numbering = hmatrix.to_dense_in_user_numbering()

    # HMatrix vector product
    dtype = np.float64 if is_complex is False else np.complex128
    np.random.seed(0)
    if is_complex is False:
        x = np.random.rand(nb_cols)
    else:
        x = np.random.rand(nb_cols) + 1j * np.random.rand(nb_cols)
    y = hmatrix * x
    y_exact = generator.mat_vec(x)
    y_dense = dense_in_user_numbering.dot(x)
    y_copy = copy_hmatrix * x
    assert np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact) < epsilon
    assert np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense) < 1e-10
    assert np.linalg.norm(y - y_copy) / np.linalg.norm(y) < 1e-10

    # HMatrix matrix product
    np.random.seed(0)
    if is_complex is False:
        x = np.random.rand(nb_cols, 2)
    else:
        x = np.random.rand(nb_cols, 2) + 1j * np.random.rand(nb_cols, 2)
    y = hmatrix @ x
    y_exact = generator.mat_mat(x)
    y_dense = dense_in_user_numbering @ x
    y_copy = copy_hmatrix @ x
    assert np.linalg.norm(y - y_exact) / np.linalg.norm(y_exact) < epsilon
    assert np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense) < 1e-10
    assert np.linalg.norm(y - y_copy) / np.linalg.norm(y) < 1e-10

    if symmetry != "N":
        # HLU factorization
        if policy:
            copy_hmatrix.lu_factorization(policy)
        else:
            copy_hmatrix.lu_factorization()

        # HLU solve vec
        x_ref = np.ones(nb_cols, dtype=dtype)
        y = hmatrix * x_ref
        x = copy_hmatrix.lu_solve("N", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # HLU solve mat
        x_ref = np.ones((nb_cols, 2), dtype=dtype)
        y = hmatrix @ x_ref
        x = copy_hmatrix.lu_solve("N", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # Cholesky factorization
        copy_hmatrix = copy.deepcopy(hmatrix)
        if policy:
            copy_hmatrix.cholesky_factorization(policy, "L")
        else:
            copy_hmatrix.cholesky_factorization("L")

        # Cholesky solve vec
        x_ref = np.ones(nb_cols, dtype=dtype)
        y = hmatrix * x_ref
        x = copy_hmatrix.cholesky_solve("L", y)
        assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < epsilon

        # Cholesky solve mat
        x_ref = np.ones((nb_cols, 2), dtype=dtype)
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
