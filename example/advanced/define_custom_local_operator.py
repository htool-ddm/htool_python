import numpy as np

import Htool


class CustomRestrictedGlobalToLocalOperator(Htool.RestrictedGlobalToLocalOperator):
    def __init__(
        self,
        generator: Htool.VirtualGenerator,
        target_local_renumbering: Htool.LocalRenumbering,
        source_local_renumbering: Htool.LocalRenumbering,
        target_use_permutation_to_mvprod: bool = False,
        source_use_permutation_to_mvprod: bool = False,
    ) -> None:
        super().__init__(
            target_local_renumbering,
            source_local_renumbering,
            target_use_permutation_to_mvprod,
            source_use_permutation_to_mvprod,
        )
        self.data = np.zeros(
            (target_local_renumbering.size, source_local_renumbering.size)
        )
        generator.build_submatrix(
            target_local_renumbering.permutation[
                target_local_renumbering.offset : target_local_renumbering.offset
                + target_local_renumbering.size
            ],
            source_local_renumbering.permutation[
                source_local_renumbering.offset : source_local_renumbering.offset
                + source_local_renumbering.size
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


class CustomLocalToLocalOperator(Htool.VirtualLocalToLocalOperator):
    def __init__(
        self,
        generator: Htool.VirtualGenerator,
        target_local_renumbering: Htool.LocalRenumbering,
        source_local_renumbering: Htool.LocalRenumbering,
    ) -> None:
        super().__init__(
            target_local_renumbering,
            source_local_renumbering,
        )
        self.data = np.zeros(
            (target_local_renumbering.size, source_local_renumbering.size)
        )
        generator.build_submatrix(
            target_local_renumbering.permutation[
                target_local_renumbering.offset : target_local_renumbering.offset
                + target_local_renumbering.size
            ],
            source_local_renumbering.permutation[
                source_local_renumbering.offset : source_local_renumbering.offset
                + source_local_renumbering.size
            ],
            self.data,
        )

    def local_add_vector_product(
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

    def local_add_matrix_product_row_major(
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
