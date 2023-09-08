import mpi4py
import numpy as np

import Htool


class CustomGenerator(Htool.VirtualGenerator):
    def __init__(self, target_cluster, target_points, source_cluster, source_points):
        super().__init__()
        self.target_points = target_points
        self.target_permutation = target_cluster.get_permutation()
        self.source_points = source_points
        self.source_permutation = source_cluster.get_permutation()
        self.nb_rows = len(self.target_permutation)
        self.nb_cols = len(self.source_permutation)

    def get_coef(self, i, j):
        return 1.0 / (
            1e-1
            + np.linalg.norm(
                self.target_points[:, self.target_permutation[i]]
                - self.source_points[:, self.source_permutation[j]]
            )
        )

    def build_submatrix(self, row_offset, col_offset, mat):
        for j in range(0, mat.shape[0]):
            for k in range(0, mat.shape[1]):
                mat[j, k] = 1.0 / (
                    1.0e-1
                    + np.linalg.norm(
                        self.target_points[:, self.target_permutation[j + row_offset]]
                        - self.source_points[:, self.source_permutation[k + col_offset]]
                    )
                )

    def mat_vec(self, x):
        y = np.zeros(self.nb_rows)
        for i in range(0, self.nb_rows):
            for j in range(0, self.nb_cols):
                y[self.target_permutation[i]] += (
                    self.get_coef(i, j) * x[self.source_permutation[j]]
                )
        return y

    def mat_mat(self, X):
        Y = np.zeros((self.nb_rows, X.shape[1]))

        for i in range(0, self.nb_rows):
            for j in range(0, X.shape[1]):
                for k in range(0, self.nb_cols):
                    Y[self.target_permutation[i], j] += (
                        self.get_coef(i, k) * X[self.source_permutation[k], j]
                    )
        return Y


class CustomGeneratorWithPermutation(Htool.VirtualGeneratorWithPermutation):
    def __init__(
        self, target_permutation, target_points, source_permutation, source_points
    ):
        super().__init__(target_permutation, source_permutation)
        self.target_points = target_points
        self.source_points = source_points
        self.nb_rows = len(target_permutation)
        self.nb_cols = len(source_permutation)

    def get_coef(self, i, j):
        return 1.0 / (
            1e-1 + np.linalg.norm(self.target_points[:, i] - self.source_points[:, j])
        )

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = 1.0 / (
                    1e-1
                    + np.linalg.norm(
                        self.target_points[:, J[j]] - self.source_points[:, K[k]]
                    )
                )

    def mat_vec(self, x):
        y = np.zeros(self.nb_rows)
        for i in range(0, self.nb_rows):
            for j in range(0, self.nb_cols):
                y[i] += self.get_coef(i, j) * x[j]
        return y

    def mat_mat(self, X):
        Y = np.zeros((self.nb_rows, X.shape[1]))

        for i in range(0, self.nb_rows):
            for j in range(0, X.shape[1]):
                for k in range(0, self.nb_cols):
                    Y[i, j] += self.get_coef(i, k) * X[k, j]
        return Y
