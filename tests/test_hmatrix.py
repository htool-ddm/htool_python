import Htool
import numpy as np
import mpi4py
import pytest

class GeneratorSubMatrix(Htool.IMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(points_target.shape[1],points_source.shape[1])
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[:, i] - self.points_source[:, j]))

    def build_submatrix(self, J , K, mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = 1.0 / (1.e-5 + np.linalg.norm(self.points_target[:,J[j]] - self.points_source[:, K[k]])) 

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



@pytest.mark.parametrize("NbRows,NbCols,Symmetric,UPLO", [
    (500, 500, 'S','L'),
    (500, 500, 'S','U'),
    (500, 500, 'N', 'N'),
    (500, 250, 'N', 'N'),
])
def test_HMatrix(NbRows, NbCols, Symmetric,UPLO):


    # Random geometry
    np.random.seed(0)
    points_target=np.zeros((3,NbRows))
    points_target[0,:] = np.random.random(NbRows)
    points_target[1,:] = np.random.random(NbRows)
    points_target[2,:] = 1

    if NbRows==NbCols:
        points_source=points_target
    else:
        points_source=np.zeros((3,NbCols))
        points_source[0,:] = np.random.random(NbCols)
        points_source[1,:] = np.random.random(NbCols)
        points_source[2,:] = 0
    
    epsilon = 1e-3
    eta = 1 
    minclustersize=10

    Generator = GeneratorSubMatrix(points_target, points_source)

    # Build
    cluster_target = Htool.PCARegularClustering(3)
    cluster_source = Htool.PCARegularClustering(3)
    cluster_target.set_minclustersize(minclustersize)
    cluster_target.build(NbRows,points_target,2)

    if Symmetric!='N':
        cluster_source = cluster_target
    else:
        cluster_source.set_minclustersize(minclustersize)
        cluster_source.build(NbCols,points_source,2)

    HMatrix = Htool.HMatrix(cluster_target,cluster_source,epsilon,eta,Symmetric,UPLO)

    if Symmetric!='N':
        HMatrix.build(Generator, points_target)
    else:
        HMatrix.build(Generator, points_target, points_source)

    # Getters
    assert HMatrix.shape[0] == NbRows
    assert HMatrix.shape[1] == NbCols
    assert len(HMatrix.get_perm_t())==NbRows
    assert len(HMatrix.get_perm_s())==NbCols

    MasterOffset_t = HMatrix.get_MasterOffset_t() 
    MasterOffset_s = HMatrix.get_MasterOffset_s() 
    assert len(MasterOffset_t)==mpi4py.MPI.COMM_WORLD.Get_size()
    assert len(MasterOffset_s)==mpi4py.MPI.COMM_WORLD.Get_size()
    assert sum([pair[1] for pair in MasterOffset_t])==NbRows
    assert sum([pair[1] for pair in MasterOffset_s])==NbCols

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

    # Test display functions
    HMatrix.display(False)
    HMatrix.display_cluster(points_target,2,"target",False)
    if Symmetric=='N':
        HMatrix.display_cluster(points_source,2,"source",False)

    # Print information
    HMatrix.print_infos()
    print(HMatrix)
    assert mpi4py.MPI.COMM_WORLD.Get_size()==int(HMatrix.get_infos("Number_of_MPI_tasks"))