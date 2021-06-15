import Htool
import numpy as np
import mpi4py

# Custom generator
class Generator(Htool.IMatrix):

    def __init__(self,points_target,points_source):
        super().__init__(points_target.shape[1],points_source.shape[1])
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[:,i] - self.points_source[:,j]))

    def build_submatrix(self, J , K, mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = 1.0 / (1.e-5 + np.linalg.norm(self.points_target[:,J[j]] - self.points_source[:,K[k]])) 

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

# Htool parameters
eta = 10
epsilon = 1e-3
minclustersize = 10

# Random geometry
NbRows = 500
NbCols = 500
np.random.seed(0)
points_target=np.zeros((2,NbRows))
sizeworld = mpi4py.MPI.COMM_WORLD.Get_size()
local_size=int(NbRows/sizeworld)
MasterOffset=np.zeros((2,sizeworld))
for i in range(0,sizeworld-1):
    MasterOffset[0,i]=i*local_size
    MasterOffset[1,i]=local_size
    points_target[0,i*local_size:(i+1)*local_size] = i

points_target[0,(sizeworld-1)*local_size:] = sizeworld-1
MasterOffset[0,sizeworld-1]=(sizeworld-1)*local_size
MasterOffset[1,sizeworld-1]=NbRows-(sizeworld-1)*local_size

points_target[1,:] = np.random.random(NbRows)

# Cluster target
cluster_target = Htool.PCARegularClustering(2)
cluster_target.set_minclustersize(minclustersize)
cluster_target.build(NbRows,points_target,MasterOffset,2)

if NbRows==NbCols:
    points_source=points_target
    cluster_source=cluster_target
else:
    points_source=np.zeros((2,NbCols))
    points_source[0,:] = np.random.random(NbCols)
    points_source[1,:] = np.random.random(NbCols)
    cluster_source = None



# Build H matrix
generator = Generator(points_target,points_source)
HMatrix_test = Htool.HMatrix(cluster_target,cluster_source,epsilon,eta)
HMatrix_test.build(generator,points_target,points_source)

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
