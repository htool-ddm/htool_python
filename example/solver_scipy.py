import Htool
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import gmres

# Custom generator
class Generator(Htool.IMatrix):

    def __init__(self,points):
        super().__init__(points.shape[1],points.shape[1])
        self.points=points

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points[:, i] - self.points[:, j]))

    def build_submatrix(self, J , K, mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = 1.0 / (1.e-5 + np.linalg.norm(self.points[:,J[j]] - self.points[: ,K[k]])) 

    def mat_vec(self,x):
        y = np.zeros(self.nb_rows())
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i,j)*x[j]
        return y



# SETUP
# n² points on a regular grid in a square
n = int(np.sqrt(4761))
points = np.zeros((2,n*n))
for j in range(0, n):
    for k in range(0, n):
        points[:,j+k*n] = (j, k)

# Htool parameters
eta = 10
epsilon = 1e-3
minclustersize = 10

# Build H matrix
generator = Generator(points)
symmetric = 'S'
UPLO = 'L'
HMatrix = Htool.HMatrix(2,epsilon,eta,symmetric,UPLO)
HMatrix.set_minclustersize(minclustersize)
HMatrix.build(generator,points)

# Dense matrix
Full_H = 1.0 / (1e-5 + norm(points.T.reshape(1, n*n, 2) - points.T.reshape(n*n, 1, 2), axis=2))

# GMRES
y = np.ones((n*n,))
x, _ = gmres(HMatrix, y)
x_full, _ = gmres(Full_H, y)

err_gmres_hmat  = norm(HMatrix @ x - y)
err_gmres_dense = norm(Full_H @ x_full - y)
err_comp        = norm(x - x_full)/norm(x)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank==0:
    print("Error from gmres (Hmatrix):", err_gmres_hmat)
    print("Error from gmres (full matrix):", err_gmres_dense)
    print("Error between the two solutions:", err_comp)

# Several ways to display information
print(HMatrix)
HMatrix.print_infos()
HMatrix.display()
HMatrix.display_cluster(points,2,"target")
