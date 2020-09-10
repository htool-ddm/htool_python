import Htool
import numpy as np
import mpi4py
import pytest


class GeneratorCoef(Htool.IMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(len(points_target),len(points_source))
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[i, :] - self.points_source[j, :]))

    def matvec(self,x):
        y = np.zeros(self.nb_rows())
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i,j)*x[j]
        return y

    def matmat(self,X):
        Y = np.zeros((self.nb_rows(), X.shape[1]))

        for i in range(0,self.nb_rows()):
            for j in range(0,X.shape[1]):
                for k in range(0,self.nb_cols()):
                    Y[i,j]+=self.get_coef(i, k)*X[k,j]
        return Y
class GeneratorSubMatrix(Htool.IMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(len(points_target),len(points_source))
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[i, :] - self.points_source[j, :]))

    def get_submatrix(self, J , K):
        submat = np.zeros((len(J),len(K)),order="C")
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                submat[j,k] = 1.0 / (1.e-5 + np.linalg.norm(self.points_target[J[j],:] - self.points_source[K[k], :])) 
        return Htool.SubMatrix(J,K,submat)

    def matvec(self,x):
        y = np.zeros(self.nb_rows())
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i, j)*x[j]
        return y

    def matmat(self,X):
        Y = np.zeros((self.nb_rows(), X.shape[1]))

        for i in range(0,self.nb_rows()):
            for j in range(0,X.shape[1]):
                for k in range(0,self.nb_cols()):
                    Y[i,j]+=self.get_coef(i,k)*X[k,j]
        return Y

def FactoryGenerator(GeneratorType, points_target, points_source):
    if GeneratorType == "Coef":
        return GeneratorCoef(points_target, points_source)
    elif GeneratorType == "SubMatrix":
        return GeneratorSubMatrix(points_target, points_source)

def FactoryHMatrix(Generator, points_target, points_source, Symmetric,UPLO):
    if Symmetric!='N':
        return Htool.HMatrix(Generator, points_target, 'N')
    else:
        return Htool.HMatrix(Generator, points_target, points_source)


@pytest.mark.parametrize("GeneratorType,NbRows,NbCols,Symmetric,UPLO", [
    ("Coef",500, 500, 'S','L'),
    ("Coef",500, 500, 'S','U'),
    ("Coef",500, 500, 'N', 'N'),
    ("Coef",500, 250, 'N', 'N'),
    ("SubMatrix",500, 500, 'S','L'),
    ("SubMatrix",500, 500, 'S','U'),
    ("SubMatrix",500, 500, 'N', 'N'),
    ("SubMatrix",500, 250, 'N', 'N'),
])
def test_HMatrix(GeneratorType, NbRows, NbCols, Symmetric,UPLO):


    # Random geometry
    np.random.seed(0)
    points_target=np.zeros((NbRows,3))
    points_target[:,0] = np.random.random(NbRows)
    points_target[:,1] = np.random.random(NbRows)
    points_target[:,2] = 1

    if NbRows==NbCols:
        points_source=points_target
    else:
        points_source=np.zeros((NbCols,3))
        points_source[:,0] = np.random.random(NbCols)
        points_source[:,1] = np.random.random(NbCols)
        points_source[:,2] = 0
    
    epsilon = 1e-3
    Htool.SetEta(1)
    Htool.SetEpsilon(epsilon)
    Htool.SetMinClusterSize(10)

    Generator = FactoryGenerator(GeneratorType, points_target, points_source)

    # Build
    HMatrix = FactoryHMatrix(Generator, points_target, points_source,Symmetric,UPLO)

    # Getters
    assert HMatrix.shape[0] == NbRows
    assert HMatrix.shape[1] == NbCols

    # Linear algebra
    x = np.random.rand(NbCols)
    y_ref = Generator.matvec(x)

    assert np.linalg.norm(HMatrix*x-y_ref)/np.linalg.norm(y_ref)<epsilon
    assert np.linalg.norm(HMatrix.matvec(x)-y_ref)/np.linalg.norm(y_ref)<epsilon
    assert np.linalg.norm(HMatrix@x-y_ref)/np.linalg.norm(y_ref)<epsilon

    X = np.random.rand(NbCols,3)
    Y_ref = Generator.matmat(X)
    Y = HMatrix@X
    assert np.linalg.norm(HMatrix@X-Y_ref)/np.linalg.norm(Y_ref)<epsilon


    # Print information
    HMatrix.print_infos()
    print(HMatrix)