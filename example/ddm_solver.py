import Htool
import numpy as np
import mpi4py

# MPI
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

# Custom generator
class Generator(Htool.VirtualGenerator):

    def __init__(self,points_target,points_source):
        super().__init__(points_target.shape[1],points_source.shape[1])
        self.points_target=points_target
        self.points_source=points_source

    def get_coef(self, i , j):
        return 1.0 / (1e-5 + np.linalg.norm(self.points_target[i, :] - self.points_source[j, :]))

    def build_submatrix(self, J , K, mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = 1.0 / (1.e-5 + np.linalg.norm(self.points_target[:, J[j]] - self.points_source[:, K[k]])) 

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
Size = 500
np.random.seed(0)
points_target=np.zeros((2,Size))
points_target[0,:] = np.random.random(Size)
points_target[1,:] = np.random.random(Size)

# Htool parameters
eta = 5
epsilon = 1e-3
minclustersize = 10

# Build H matrix
generator = Generator(points_target,points_target)
cluster = Htool.PCARegularClustering(2)
cluster.build(Size,points_target,2)
cluster.set_minclustersize(minclustersize)
hmat = Htool.HMatrix(cluster,cluster,epsilon,eta,'S','L')
hmat.build(generator,points_target)

# Solver with block Jacobi
x_ref = np.random.random(Size)
b = hmat*x_ref
x =np.zeros(Size)
ddm_solver = Htool.DDM(hmat)

hpddm_args="-hpddm_compute_residual l2 "
if (rank==0):
    hpddm_args+="-hpddm_verbosity 10"
ddm_solver.set_hpddm_args(hpddm_args)
ddm_solver.facto_one_level()
ddm_solver.solve(x,b)

# Several ways to display information
print(hmat)
hmat.print_infos()
hmat.display()
hmat.display_cluster(points_target,2,"target")
ddm_solver.print_infos()

if rank==0:
    nb_it = ddm_solver.get_infos("Nb_it")
    print("Nb_it",nb_it)
    print(np.linalg.norm(x-x_ref)/np.linalg.norm(x_ref))

