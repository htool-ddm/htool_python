import Htool
import numpy as np
import pytest

class GeneratorSubMatrix(Htool.VirtualGenerator):
    def build_submatrix(self,J,K,mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = J[j]+K[k]


@pytest.mark.parametrize("NbRows,NbCols", [
    (10, 10),(10, 20),
])
def test_IMatrix(NbRows, NbCols):
    generator = GeneratorSubMatrix(NbRows, NbCols)
    mat = np.zeros((2,2))
    print(generator.build_submatrix(np.array([1, 2]), np.array([1, 2]),mat))
    assert generator.nb_cols() == NbCols
    assert generator.nb_rows() == NbRows
    generator.build_submatrix(np.array([1, 2]), np.array([1, 2]),mat)
    assert ( mat == np.array([[2, 3], [3, 4]])).all()
