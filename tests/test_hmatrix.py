import Htool
import numpy as np
import mpi4py
import pytest
import math
import matplotlib.pyplot as plt


class GeneratorSubMatrix(Htool.VirtualGenerator):

    def __init__(self, points_target, points_source):
        super().__init__(points_target.shape[1], points_source.shape[1])
        self.points_target = points_target
        self.points_source = points_source

    def get_coef(self, i, j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[:, i] - self.points_source[:, j]))

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = 1.0 / (1.e-5 + np.linalg.norm(
                    self.points_target[:, J[j]] - self.points_source[:, K[k]]))

    def matvec(self, x):
        y = np.zeros(self.nb_rows())
        for i in range(0, self.nb_rows()):
            for j in range(0, self.nb_cols()):
                y[i] += self.get_coef(i, j)*x[j]
        return y

    def matmat(self, X):
        Y = np.zeros((self.nb_rows(), X.shape[1]))

        for i in range(0, self.nb_rows()):
            for j in range(0, X.shape[1]):
                for k in range(0, self.nb_cols()):
                    Y[i, j] += self.get_coef(i, k)*X[k, j]
        return Y

    def inplace_matmat(self, X, Y):
        for i in range(0, self.nb_rows()):
            for j in range(0, X.shape[1]):
                for k in range(0, self.nb_cols()):
                    Y[i, j] += self.get_coef(i, k)*X[k, j]


class CustomSVD(Htool.CustomLowRankGenerator):

    def build_low_rank_approximation(self, epsilon, rank, A, J, K):
        submat = np.zeros((len(J), len(K)))
        A.build_submatrix(J, K, submat)
        u, s, vh = np.linalg.svd(submat, full_matrices=False)

        norm = np.linalg.norm(submat)
        svd_norm = 0
        count = len(s)-1
        while count > 0 and math.sqrt(svd_norm)/norm < epsilon:
            svd_norm += s[count]**2
            count -= 1
        count = min(count+1, min(len(J), len(K)))
        self.set_U(u[:, 0:count]*s[0:count])
        self.set_V(vh[0:count, :])
        self.set_rank(count)


class CustomOffDiagonalApproximation(Htool.CustomOffDiagonalApproximation):
    def __init__(self, HA, off_diagonal_points_target, off_diagonal_points_source):
        Htool.CustomOffDiagonalApproximation.__init__(self, HA)
        self.off_diagonal_generator = GeneratorSubMatrix(
            off_diagonal_points_target, off_diagonal_points_source)

    def mat_mat_prod_global_to_local(self, x, y):
        self.off_diagonal_generator.inplace_matmat(x, y)


class DenseBlockGenerator(Htool.CustomDenseBlocksGenerator):
    def __init__(self, points_target, points_source):
        super().__init__()
        self.points_target = points_target
        self.points_source = points_source
        self.points_target = points_target
        self.points_source = points_source

    def build_dense_blocks(self, rows, cols, blocks):
        nb_blocks = len(rows)  # =len(cols)=len(blocks)
        for i in range(nb_blocks):
            J, K = blocks[i].shape
            for j in range(J):
                for k in range(0, K):
                    blocks[i][j, k] = 1.0 / (1.e-5 + np.linalg.norm(
                        self.points_target[:, rows[i][j]] - self.points_source[:, cols[i][k]]))


@pytest.mark.parametrize("NbRows,NbCols,Symmetric,UPLO,Compression,Delay,OffDiagonalApproximation", [
    (500, 500, 'S', 'L', None, False, None),
    (500, 500, 'S', 'U', None, False, None),
    (500, 500, 'N', 'N', None, False, None),
    (500, 250, 'N', 'N', None, False, None),
    (500, 500, 'S', 'L', None, True, None),
    (500, 500, 'S', 'U', None, True, None),
    (500, 500, 'N', 'N', None, True, None),
    (500, 500, 'S', 'L', "Custom", True, None),
    (500, 500, 'S', 'U', "Custom", True, None),
    (500, 500, 'N', 'N', "Custom", True, None),
    (500, 250, 'N', 'N', "Custom", True, None),
    (500, 500, 'S', 'L', None, False, "Dense"),
    (500, 500, 'S', 'U', None, False, "Dense"),
    (500, 500, 'N', 'N', None, False, "Dense"),
    (500, 250, 'N', 'N', None, False, "Dense"),
    (500, 500, 'S', 'L', None, False, "HMatrix"),
    (500, 500, 'S', 'U', None, False, "HMatrix"),
    (500, 500, 'N', 'N', None, False, "HMatrix"),
    (500, 250, 'N', 'N', None, False, "HMatrix"),
    (500, 500, 'S', 'L', "Custom", False, "Dense"),
    (500, 500, 'S', 'U', "Custom", False, "Dense"),
    (500, 500, 'N', 'N', "Custom", False, "Dense"),
    (500, 250, 'N', 'N', "Custom", False, "Dense"),
    (500, 500, 'S', 'L', "Custom", False, "HMatrix"),
    (500, 500, 'S', 'U', "Custom", False, "HMatrix"),
    (500, 500, 'N', 'N', "Custom", False, "HMatrix"),
    (500, 250, 'N', 'N', "Custom", False, "HMatrix"),
])
def test_HMatrix(NbRows, NbCols, Symmetric, UPLO, Compression, Delay, OffDiagonalApproximation):

    # Random geometry
    np.random.seed(0)
    points_target = np.zeros((3, NbRows))
    points_target[0, :] = np.random.random(NbRows)
    points_target[1, :] = np.random.random(NbRows)
    points_target[2, :] = 1

    if NbRows == NbCols:
        points_source = points_target
    else:
        points_source = np.zeros((3, NbCols))
        points_source[0, :] = np.random.random(NbCols)
        points_source[1, :] = np.random.random(NbCols)
        points_source[2, :] = 0

    epsilon = 1e-3
    eta = 1
    minclustersize = 10

    Generator = GeneratorSubMatrix(points_target, points_source)

    # Build
    cluster_target = Htool.PCARegularClustering(3)
    cluster_source = Htool.PCARegularClustering(3)
    cluster_target.set_minclustersize(minclustersize)
    cluster_target.build(NbRows, points_target, 2)

    if Symmetric != 'N':
        cluster_source = cluster_target
    else:
        cluster_source.set_minclustersize(minclustersize)
        cluster_source.build(NbCols, points_source, 2)

    HMatrix = Htool.HMatrix(cluster_target, cluster_source,
                            epsilon, eta, Symmetric, UPLO)

    if OffDiagonalApproximation is not None:
        # Geometry
        off_diagonal_points_target, off_diagonal_points_source = HMatrix.get_off_diagonal_geometries(
            points_target, points_source)

        if OffDiagonalApproximation == "Dense":
            off_diagonal_approximation = CustomOffDiagonalApproximation(HMatrix,
                                                                        off_diagonal_points_target, off_diagonal_points_source)

        if OffDiagonalApproximation == "HMatrix":
            # Clustering
            off_diagonal_cluster_target = Htool.PCARegularClustering(3)
            off_diagonal_cluster_source = Htool.PCARegularClustering(3)
            off_diagonal_cluster_target.set_minclustersize(minclustersize)
            off_diagonal_cluster_source.set_minclustersize(minclustersize)
            off_diagonal_cluster_target.build(
                off_diagonal_points_target.shape[1], off_diagonal_points_target, 2, mpi4py.MPI.COMM_SELF)
            off_diagonal_cluster_source.build(
                off_diagonal_points_source.shape[1], off_diagonal_points_source, 2, mpi4py.MPI.COMM_SELF)

            # Off diagonal generator
            off_diagonal_generator = GeneratorSubMatrix(
                off_diagonal_points_target, off_diagonal_points_source)

            # Off diagonal HMatrix
            off_diagonal_approximation = Htool.HMatrixOffDiagonalApproximation(
                HMatrix, off_diagonal_cluster_target, off_diagonal_cluster_source)
            if Compression is not None:
                compression = CustomSVD()
                off_diagonal_approximation.set_compression(compression)

            off_diagonal_approximation.build(off_diagonal_generator,
                                             off_diagonal_points_target, off_diagonal_points_source)

            off_diagonal_approximation.print_infos()

        HMatrix.set_off_diagonal_approximation(
            off_diagonal_approximation)

    if Compression is not None:
        compression = CustomSVD()
        HMatrix.set_compression(compression)

    HMatrix.set_delay_dense_computation(Delay)

    if Symmetric != 'N':
        HMatrix.build(Generator, points_target)
    else:
        HMatrix.build(Generator, points_target, points_source)

    if Delay:
        dense_blocks_generator = DenseBlockGenerator(
            points_target, points_source)
        HMatrix.build_dense_blocks(dense_blocks_generator)

    # Getters
    assert HMatrix.shape[0] == NbRows
    assert HMatrix.shape[1] == NbCols
    assert len(HMatrix.get_perm_t()) == NbRows
    assert len(HMatrix.get_perm_s()) == NbCols

    MasterOffset_t = HMatrix.get_MasterOffset_t()
    MasterOffset_s = HMatrix.get_MasterOffset_s()
    assert len(MasterOffset_t) == mpi4py.MPI.COMM_WORLD.Get_size()
    assert len(MasterOffset_s) == mpi4py.MPI.COMM_WORLD.Get_size()
    assert sum([pair[1] for pair in MasterOffset_t]) == NbRows
    assert sum([pair[1] for pair in MasterOffset_s]) == NbCols

    # Linear algebra
    x = np.random.rand(NbCols)
    y_ref = Generator.matvec(x)

    assert np.linalg.norm(HMatrix*x-y_ref)/np.linalg.norm(y_ref) < epsilon
    assert np.linalg.norm(HMatrix.matvec(x)-y_ref) / \
        np.linalg.norm(y_ref) < epsilon
    assert np.linalg.norm(HMatrix@x-y_ref)/np.linalg.norm(y_ref) < epsilon

    X = np.random.rand(NbCols, 3)
    Y_ref = Generator.matmat(X)
    Y = HMatrix@X
    assert np.linalg.norm(HMatrix@X-Y_ref)/np.linalg.norm(Y_ref) < epsilon

    # Test display functions
    HMatrix.display(False)
    HMatrix.display_cluster(points_target, 2, "target", False)
    if Symmetric == 'N':
        HMatrix.display_cluster(points_source, 2, "source", False)
    plt.close()

    # Print information
    HMatrix.print_infos()
    print(HMatrix)
    assert mpi4py.MPI.COMM_WORLD.Get_size() == int(
        HMatrix.get_infos("Number_of_MPI_tasks"))
