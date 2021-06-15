import Htool
import numpy as np
from mpi4py import MPI
import math
import struct
import os
import mpi4py
import pytest


class GeneratorCoef(Htool.ComplexIMatrix):
    def __init__(self,matrix):
        super().__init__(matrix.shape[0],matrix.shape[1])
        self.matrix=matrix

    def get_coef(self, i , j):
        return self.matrix[i,j]

    def build_submatrix(self, J , K, mat):
        for j in range(0,len(J)):
            for k in range(0,len(K)):
                mat[j,k] = self.get_coef(J[j],K[k])

@pytest.mark.parametrize("mu,Symmetry", [
    (1, 'S'),
    (10, 'S'),
    (1, 'N'),
    (10, 'N'),
])
def test_ddm_solver(mu,Symmetry):

    # MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Htool
    tol = 1e-6
    eta = 0.1

    # args
    folder = "non_symmetric"
    UPLO='N'
    if Symmetry=='S':
        folder = "symmetric"
        UPLO='L'

    # Matrix
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/matrix.bin"), "rb" ) as input:
        data=input.read()
        (m, n) = struct.unpack("@II", data[:8])
        # print(m,n)
        A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
        A=np.transpose(A.reshape((m,n)))

    # Right-hand side
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/rhs.bin"), "rb" ) as input:
        data=input.read()
        l = struct.unpack("@I", data[:4])
        rhs=np.frombuffer(data[4:],dtype=np.dtype('complex128'))
    f = np.zeros(len(rhs),dtype="complex128")
    if mu>1:
        f = np.zeros((len(rhs),mu),dtype="complex128")
        for p in range(0,mu):
            f[:,p]=rhs
    else:
        f = rhs

    # mesh
    p=np.zeros((3,n))
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/mesh.msh"), "r" ) as input:
        check=False
        count=0
        for line in input:

            if line=="$EndNodes\n":
                break

            if check and len(line.split())==4:
                tab_line=line.split()
                p[0][count]=tab_line[1]
                p[1][count]=tab_line[2]
                p[2][count]=tab_line[3]
                count+=1

            if line=="$Nodes\n":
                check=True

    # Cluster
    cluster = Htool.PCARegularClustering(3)
    cluster.read_cluster(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/cluster_"+str(size)+"_permutation.csv"),os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/cluster_"+str(size)+"_tree.csv"))

    # Hmatrix
    generator = GeneratorCoef(A)
    hmat = Htool.ComplexHMatrix(cluster,cluster,tol,eta,Symmetry,UPLO)
    hmat.build(generator,p)

    # Global vectors
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/sol.bin"), "rb" ) as input:
        data=input.read()
        x_ref = np.frombuffer(data[4:],dtype=np.dtype('complex128'))
    x =np.zeros(len(f),dtype="complex128", order="F")
    if mu>1:
        x =np.zeros((len(f),mu),dtype="complex128", order="F")

    # Domain decomposition
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/cluster_to_ovr_subdomain_"+str(size)+"_"+str(rank)+".bin"), "rb" ) as input:
        data=input.read()
        cluster_to_ovr_subdomain = np.frombuffer(data[4:],dtype=np.dtype('int32'))
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/ovr_subdomain_to_global_"+str(size)+"_"+str(rank)+".bin"), "rb" ) as input:
        data=input.read()
        ovr_subdomain_to_global = np.frombuffer(data[4:],dtype=np.dtype('int32'))
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/neighbors_"+str(size)+"_"+str(rank)+".bin"), "rb" ) as input:
        data=input.read()
        neighbors = np.frombuffer(data[4:],dtype=np.dtype('int32'))
    

    intersections = []
    for p in range(0,len(neighbors)):
        with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/"+folder+"/intersections_"+str(size)+"_"+str(rank)+"_"+str(p)+".bin"), "rb" ) as input:
            data=input.read()
            intersection = np.frombuffer(data[4:],dtype=np.dtype('int32'))
            intersections.append(intersection)

    # Solvers
    block_jacobi = Htool.ComplexDDM(hmat)
    DDM_solver   = Htool.ComplexDDM(generator,hmat,ovr_subdomain_to_global,cluster_to_ovr_subdomain,neighbors,intersections)


    # No precond wo overlap
    if rank==0:
        print("No precond without overlap:")
    block_jacobi.solve(x,f,hpddm_args="-hpddm_schwarz_method none -hpddm_max_it 200 -hpddm_tol "+str(tol))
    block_jacobi.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # DDM one level ASM wo overlap
    if rank==0:
        print("ASM one level without overlap:")
    comm.Barrier()
    block_jacobi.set_hpddm_args("-hpddm_schwarz_method asm")
    block_jacobi.facto_one_level()
    block_jacobi.solve(x,f)
    block_jacobi.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # DDM one level ASM wo overlap
    if rank==0:
        print("RAS one level without overlap:")
    comm.Barrier()
    block_jacobi.set_hpddm_args("-hpddm_schwarz_method ras")
    block_jacobi.solve(x,f)
    block_jacobi.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # Check infos
    if (mpi4py.MPI.COMM_WORLD.Get_rank()==0):
        assert mpi4py.MPI.COMM_WORLD.Get_size()==int(block_jacobi.get_infos("Nb_subdomains"))

    # No precond with overlap
    if rank==0:
        print("No precond with overlap:")
    DDM_solver.solve(x,f,hpddm_args="-hpddm_schwarz_method none -hpddm_max_it 200 -hpddm_tol "+str(tol))
    DDM_solver.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # DDM one level ASM with overlap
    if rank==0:
        print("ASM one level with overlap:")
    comm.Barrier()
    DDM_solver.set_hpddm_args("-hpddm_schwarz_method asm")
    DDM_solver.facto_one_level()
    DDM_solver.solve(x,f)
    DDM_solver.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # DDM one level RAS with overlap
    if rank==0:
        print("RAS one level with overlap:")
    comm.Barrier()
    DDM_solver.set_hpddm_args("-hpddm_schwarz_method ras")
    DDM_solver.solve(x,f)
    DDM_solver.print_infos()
    if mu==1:
        error = np.linalg.norm(hmat*x-f)/np.linalg.norm(f)
    elif mu>1:
        error = np.linalg.norm(hmat@x-f)/np.linalg.norm(f)
    if rank==0:
        print(error)
    assert error < tol
    x.fill(0)

    # Check infos
    if (mpi4py.MPI.COMM_WORLD.Get_rank()==0):
        assert mpi4py.MPI.COMM_WORLD.Get_size()==int(DDM_solver.get_infos("Nb_subdomains"))