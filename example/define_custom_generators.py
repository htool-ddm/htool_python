import Htool
import numpy as np


class CustomGenerator(Htool.VirtualGeneratorInUserNumbering):
    def __init__(self, target_points, source_points):
        super().__init__()
        self.target_points = target_points
        self.source_points = source_points
        self.nb_rows = target_points.shape[1]
        self.nb_cols = source_points.shape[1]

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
