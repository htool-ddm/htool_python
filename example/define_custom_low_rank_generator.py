import math

import Htool
import numpy as np


class CustomSVD(Htool.VirtualLowRankGenerator):
    def build_low_rank_approximation(
        self, generator, target_size, source_size, target_offset, source_offset, epsilon
    ):
        submat = np.zeros((target_size, source_size), order="F")
        generator.build_submatrix(target_offset, source_offset, submat)
        u, s, vh = np.linalg.svd(submat, full_matrices=False)

        norm = np.linalg.norm(submat)
        svd_norm = 0
        count = len(s) - 1
        while count > 0 and math.sqrt(svd_norm) / norm < epsilon:
            svd_norm += s[count] ** 2
            count -= 1
        count = min(count + 1, min(target_size, source_size))
        self.set_U(u[:, 0:count] * s[0:count])
        self.set_V(vh[0:count, :])
        self.set_rank(count)
