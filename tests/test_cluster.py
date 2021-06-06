import Htool
import numpy as np
from mpi4py import MPI
import math
import struct
import os


def test_cluster():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Matrix
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/non_symmetric/matrix.bin"), "rb" ) as input:
        data=input.read()
        (m, n) = struct.unpack("@II", data[:8])
        # print(m,n)
        A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
        A=np.transpose(A.reshape((m,n)))

    # mesh
    p=np.zeros((3,n))
    with open(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/non_symmetric/mesh.msh"), "r" ) as input:
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
    cluster = Htool.Cluster(3)
    cluster.read_cluster(os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/non_symmetric/cluster_"+str(size)+"_permutation.csv"),os.path.join(os.path.dirname(__file__)+"/../lib/htool/data/data_test/non_symmetric/cluster_"+str(size)+"_tree.csv"))

