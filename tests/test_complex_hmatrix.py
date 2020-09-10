import Htool
import numpy as np
import mpi4py
import pytest


class GeneratorCoef(Htool.ComplexIMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(len(points_target),len(points_source))
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):

        return (1.0+1j*np.sign(i-j)) / (1e-5 + np.linalg.norm(self.points_target[i, :] - self.points_source[j, :]))

    def matvec(self,x):
        y = np.zeros(self.nb_rows(),dtype="complex128")
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i,j)*x[j]
        return y

    def matmat(self,X):
        Y = np.zeros((self.nb_rows(), X.shape[1]),dtype="complex128")

        for i in range(0,self.nb_rows()):
            for j in range(0,X.shape[1]):
                for k in range(0,self.nb_cols()):
                    Y[i,j]+=self.get_coef(i, k)*X[k,j]
        return Y
    
    def print(self):
        matrix = np.zeros((self.nb_rows(),self.nb_cols()),dtype="complex128")
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                matrix[i,j]=self.get_coef(i,j)

        print(matrix)
class GeneratorSubMatrix(Htool.ComplexIMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(len(points_target),len(points_source))
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return (1.0+1j*np.sign(i-j)) / (1e-5 + np.linalg.norm(self.points_target[i, :] - self.points_source[j, :]))

    def get_submatrix(self, J , K):
        submat = np.zeros((len(J),len(K)),order="C",dtype="complex128")
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                submat[j,k] = (1.0+1j*np.sign(J[j]-K[k])) / (1.e-5 + np.linalg.norm(self.points_target[J[j],:] - self.points_source[K[k], :])) 
        return Htool.ComplexSubMatrix(J,K,submat)

    def matvec(self,x):
        y = np.zeros(self.nb_rows(),dtype="complex128")
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i, j)*x[j]
        return y

    def matmat(self,X):
        Y = np.zeros((self.nb_rows(), X.shape[1]),dtype="complex128")

        for i in range(0,self.nb_rows()):
            for j in range(0,X.shape[1]):
                for k in range(0,self.nb_cols()):
                    Y[i,j]+=self.get_coef(i,k)*X[k,j]
        return Y
    def print(self):
        matrix = np.zeros((self.nb_rows(),self.nb_cols()),dtype="complex128")
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                matrix[i,j]=self.get_coef(i,j)

def FactoryGenerator(GeneratorType, points_target, points_source):
    if GeneratorType == "Coef":
        return GeneratorCoef(points_target, points_source)
    elif GeneratorType == "SubMatrix":
        return GeneratorSubMatrix(points_target, points_source)

def FactoryHMatrix(Generator, points_target, points_source, Symmetric,UPLO):
    if Symmetric!='N':
        return Htool.ComplexHMatrix(Generator, points_target, Symmetric,UPLO)
    else:
        return Htool.ComplexHMatrix(Generator, points_target, points_source)

# Interesingly partialACA does not perform well sometimes with a complex field, looking for alternative compressors
@pytest.mark.parametrize("GeneratorType,NbRows,NbCols,Symmetric,UPLO", [
    ("Coef",500, 500, 'H','L'),
    ("Coef",500, 500, 'H','U'),
    ("Coef",500, 500, 'N','N'),
    # ("Coef",500, 400, 'N','N'),
    ("SubMatrix",500, 500, 'H','L'),
    ("SubMatrix",500, 500, 'H','U'),
    ("SubMatrix",500, 500, 'N','N'),
    # ("SubMatrix",500, 400, 'N','N'),
])
def test_Complex_HMatrix(GeneratorType, NbRows, NbCols, Symmetric,UPLO):


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