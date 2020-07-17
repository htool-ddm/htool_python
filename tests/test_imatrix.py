import Htool
import numpy as np
import pytest

class GeneratorCoef(Htool.IMatrix):
    def get_coef(self, i , j):
        return i+j


class GeneratorSubMatrix(Htool.IMatrix):
    def get_coef(self, i , j):
        return i+j

    def get_submatrix(self,J,K):
        submat = np.zeros((len(J),len(K)),order="C")
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                submat[j,k] = J[j]+K[k]
        return Htool.SubMatrix(J, K, submat)


def FactoryGenerator(GeneratorType, NbRows, NbCols):
    if GeneratorType == "Coef":
        return GeneratorCoef(NbRows, NbCols)
    elif GeneratorType == "SubMatrix":
        return GeneratorSubMatrix(NbRows, NbCols)


@pytest.mark.parametrize("GeneratorType,NbRows,NbCols", [
    ("Coef",10, 10),
    ("SubMatrix",10, 10),
])
def test_IMatrix(GeneratorType, NbRows, NbCols):
    generator = FactoryGenerator(GeneratorType, NbRows, NbCols)
    print(generator.get_submatrix(np.array([1, 2]), np.array([1, 2])).get_matrix())
    assert generator.nb_cols() == NbCols
    assert generator.nb_rows() == NbRows
    assert generator.get_coef(2, 2) == 4
    assert (generator.get_submatrix(np.array([1, 2]), np.array([1, 2])).get_matrix() == np.array([[2, 3], [3, 4]])).all()
