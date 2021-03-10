import Htool
import numpy as np
import mpi4py

# MPI
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

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
Size = 500
np.random.seed(0)
points_target=np.zeros((Size,3))
points_target[:,0] = np.random.random(Size)
points_target[:,1] = np.random.random(Size)
points_target[:,2] = 1

# Htool parameters
Htool.SetEta(10)
Htool.SetEpsilon(1e-3)
Htool.SetMinClusterSize(5)

# Build H matrix
generator = Generator(points_target,points_target)
hmat = Htool.HMatrix(generator,points_target,'S','L')

# Solver with block Jacobi
x_ref = np.random.random(Size)
b = hmat*x_ref
x =np.zeros(Size)
ddm_solver = Htool.DDM(hmat)
ddm_solver.set_hpddm_args("-hpddm_verbosity 10 -hpddm_compute_residual l2")
ddm_solver.facto_one_level()
ddm_solver.solve(x,b)

# Several ways to display information
print(hmat)
hmat.print_infos()
hmat.display()
hmat.display_cluster(points_target,2,"target")
ddm_solver.print_infos()

nb_it = ddm_solver.get_infos("Nb_it")
print("Nb_it",nb_it)

if rank==0:
    print(np.linalg.norm(x-x_ref)/np.linalg.norm(x_ref))

