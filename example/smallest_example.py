import Htool
import numpy as np
import mpi4py

# Custom generator
class Generator(Htool.IMatrix):

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

    def mat_vec(self,x):
        y = np.zeros(self.nb_rows())
        for i in range(0,self.nb_rows()):
            for j in range(0,self.nb_cols()):
                y[i]+=self.get_coef(i,j)*x[j]
        return y

    def mat_mat(self,X):
        Y = np.zeros((self.nb_rows(), X.shape[1]))

        for i in range(0,self.nb_rows()):
            for j in range(0,X.shape[1]):
                for k in range(0,self.nb_cols()):
                    Y[i,j]+=self.get_coef(i, k)*X[k,j]
        return Y

# Random geometry
NbRows = 500
NbCols = 250
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

# Htool parameters
Htool.SetEta(10)
Htool.SetEpsilon(1e-3)
Htool.SetMinClusterSize(5)

# Build H matrix
generator = Generator(points_target,points_source)
HMatrix_test = Htool.HMatrix(generator,points_target,points_source)

# Test matrix vector product
x = np.random.rand(NbCols)
y_1 = HMatrix_test*x
y_2 = generator.mat_vec(x)

print(np.linalg.norm(y_1-y_2)/np.linalg.norm(y_2))

# Test matrix matrix product
X = np.random.rand(NbCols,2)
Y_1 = HMatrix_test @ X
Y_2 = generator.mat_mat(X)

print(np.linalg.norm(Y_1-Y_2)/np.linalg.norm(Y_2))

# Several ways to display information
HMatrix_test.print_infos()
print(HMatrix_test)
print("Number_of_MPI_tasks:",HMatrix_test.get_infos("Number_of_MPI_tasks"))
HMatrix_test.display()
HMatrix_test.display_cluster(points_target,2,"target")
HMatrix_test.display_cluster(points_source,2,"source")
