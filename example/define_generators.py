import numpy as np

import Htool


class CustomGenerator(Htool.VirtualGenerator):
    def __init__(self, target_points, source_points):
        super().__init__()
        self.target_points = target_points
        self.source_points = source_points
        self.nb_rows = target_points.shape[1]
        self.nb_cols = source_points.shape[1]

    def get_coef(self, i, j):
        return 1.0 / (
            1e-5 + np.linalg.norm(self.target_points[:, i] - self.source_points[:, j])
        )

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                mat[j, k] = 1.0 / (
                    1e-5
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


class ComplexCustomGenerator(Htool.ComplexVirtualGenerator):
    def __init__(self, target_points, source_points):
        super().__init__()
        self.target_points = target_points
        self.source_points = source_points
        self.nb_rows = target_points.shape[1]
        self.nb_cols = source_points.shape[1]

    def get_coef(self, i, j):
        diff = self.target_points[:, i] - self.source_points[:, j]
        sign = 1 if diff[0] > 0 else -1
        if i == j:
            sign = 0
        r = np.linalg.norm(diff)
        return (1 + sign * 1j) / (1e-5 + r)

    def build_submatrix(self, J, K, mat):
        for j in range(0, len(J)):
            for k in range(0, len(K)):
                diff = self.target_points[:, J[j]] - self.source_points[:, K[k]]
                sign = 1 if diff[0] > 0 else -1
                if J[j] == K[k]:
                    sign = 0
                r = np.linalg.norm(diff)
                mat[j, k] = (1 + sign * 1j) / (1e-5 + r)

    def mat_vec(self, x):
        y = np.zeros(self.nb_rows, dtype=np.complex128)
        for i in range(0, self.nb_rows):
            for j in range(0, self.nb_cols):
                y[i] += self.get_coef(i, j) * x[j]
        return y

    def mat_mat(self, X):
        Y = np.zeros((self.nb_rows, X.shape[1]), dtype=np.complex128)

        for i in range(0, self.nb_rows):
            for j in range(0, X.shape[1]):
                for k in range(0, self.nb_cols):
                    Y[i, j] += self.get_coef(i, k) * X[k, j]
        return Y
