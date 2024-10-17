import Htool
import numpy as np


class CustomLocalOperator(Htool.LocalOperator):
    def __init__(
        self,
        generator: Htool.VirtualGenerator,
        target_cluster: Htool.Cluster,
        source_cluster: Htool.Cluster,
        symmetry: str = "N",
        UPLO: str = "N",
        target_use_permutation_to_mvprod: bool = False,
        source_use_permutation_to_mvprod: bool = False,
    ) -> None:
        super().__init__(
            target_cluster,
            source_cluster,
            symmetry,
            UPLO,
            target_use_permutation_to_mvprod,
            source_use_permutation_to_mvprod,
        )
        self.data = np.zeros((target_cluster.get_size(), source_cluster.get_size()))
        generator.build_submatrix(
            target_cluster.get_permutation()[
                target_cluster.get_offset() : target_cluster.get_offset()
                + target_cluster.get_size()
            ],
            source_cluster.get_permutation()[
                source_cluster.get_offset() : source_cluster.get_offset()
                + source_cluster.get_size()
            ],
            self.data,
        )

    def add_vector_product(
        self, trans, alpha, input: np.array, beta, output: np.array
    ) -> None:
        # Beware, inplace operation needed for output to keep the underlying data
        output *= beta
        if trans == "N":
            output += alpha * self.data.dot(input)
        elif trans == "T":
            output += alpha * np.transpose(self.data).dot(input)
        elif trans == "C":
            output += alpha * np.vdot(np.transpose(self.data), input)

    def add_matrix_product_row_major(
        self, trans, alpha, input: np.array, beta, output: np.array
    ) -> None:
        output *= beta
        if trans == "N":
            output += alpha * self.data @ input
        elif trans == "T":
            output += alpha * np.transpose(self.data) @ input
        elif trans == "C":
            output += alpha * np.matrix.H(self.data) @ input
        output = np.asfortranarray(output)
