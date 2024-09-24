import math

import Htool
import numpy as np


class CustomSVD(Htool.VirtualLowRankGenerator):
    def __init__(self, generator: Htool.VirtualGenerator, allow_copy: bool = True):
        super().__init__(allow_copy)
        self.generator = generator

    def build_low_rank_approximation(self, rows, cols, epsilon):
        submat = np.zeros((len(rows), len(cols)), order="F")
        self.generator.build_submatrix(rows, cols, submat)
        u, s, vh = np.linalg.svd(submat, full_matrices=False)

        norm = np.linalg.norm(submat)
        svd_norm = 0
        truncated_rank = len(s) - 1
        while truncated_rank > 0 and math.sqrt(svd_norm) / norm < epsilon:
            svd_norm += s[truncated_rank] ** 2
            truncated_rank -= 1
        truncated_rank += 1

        if truncated_rank * (len(rows) + len(cols)) > (len(rows) * len(cols)):
            return False  # the low rank approximation is not worthwhile

        self.set_U(u[:, 0:truncated_rank] * s[0:truncated_rank])
        self.set_V(vh[0:truncated_rank, :])
        return True
