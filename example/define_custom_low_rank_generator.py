import math

import Htool
import numpy as np


class CustomSVD(Htool.VirtualLowRankGenerator):
    def __init__(self, generator: Htool.VirtualGenerator):
        super().__init__()
        self.generator = generator

    def build_low_rank_approximation(self, rows, cols, epsilon):
        submat = np.zeros((len(rows), len(cols)), order="F")
        self.generator.build_submatrix(rows, cols, submat)
        u, s, vh = np.linalg.svd(submat, full_matrices=False)

        norm = np.linalg.norm(submat)
        svd_norm = 0
        count = len(s) - 1
        while count > 0 and math.sqrt(svd_norm) / norm < epsilon:
            svd_norm += s[count] ** 2
            count -= 1
        count = min(count + 1, min(len(rows), len(cols)))
        self.set_U(u[:, 0:count] * s[0:count])
        self.set_V(vh[0:count, :])
