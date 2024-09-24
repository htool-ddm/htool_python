import copy
import logging

import Htool
import numpy as np
import pytest

from example.advanced.define_custom_low_rank_generator import CustomSVD
from example.create_geometry import create_random_geometries
from example.define_generators import CustomGenerator


@pytest.mark.parametrize(
    "loglevel",
    [logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL],
)
def test_hmatrix(loglevel):
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
    source_cluster: Htool.Cluster = cluster_builder.create_cluster_tree(
        source_points, number_of_children
    )

    # Build generator
    generator = CustomGenerator(target_points, source_points)

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
    y_dense = dense_in_user_numbering.dot(x)
    y_copy = copy_hmatrix * x
    assert np.linalg.norm(y - y_dense) / np.linalg.norm(y_dense) < epsilon
    assert np.linalg.norm(y - y_copy) < 1e-10

    # Output
    hmatrix_tree_parameter = hmatrix.get_tree_parameters()
    hmatrix_local_information = hmatrix.get_local_information()
    print(hmatrix_tree_parameter)
    print(hmatrix_local_information)

    # Clear low rank matrices stored in low_rank_generator when disallowing copy
    low_rank_generator.clear_data()
