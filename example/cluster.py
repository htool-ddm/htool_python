import Htool
import numpy as np
import mpi4py

# Parameters
minclustersize = 5
Number_points = 500
np.random.seed(0)
points_3D=np.zeros((3,Number_points))
points_2D=np.zeros((2,Number_points))

# Random 3D geometry
points_3D[0,:] = np.random.random(Number_points)
points_3D[1,:] = np.random.random(Number_points)
points_3D[2,:] = np.random.random(Number_points)



# Cluster 3D
cluster = Htool.Cluster(3)
cluster.set_minclustersize(minclustersize)
cluster.build(Number_points,points_3D,2)
cluster.display(points_3D,1)
cluster.display(points_3D,2)

# Cluster 3D with given partition

sizeworld = mpi4py.MPI.COMM_WORLD.Get_size()
local_size=int(Number_points/sizeworld)
MasterOffset=np.zeros((2,sizeworld))
for i in range(0,sizeworld-1):
    MasterOffset[0,i]=i*local_size
    MasterOffset[1,i]=local_size
    points_3D[0,i*local_size:(i+1)*local_size] = i
points_3D[0,(sizeworld-1)*local_size:] = sizeworld-1
MasterOffset[0,sizeworld-1]=(sizeworld-1)*local_size
MasterOffset[1,sizeworld-1]=Number_points-(sizeworld-1)*local_size


cluster = Htool.Cluster(3)
cluster.set_minclustersize(minclustersize)
cluster.build(Number_points,points_3D,MasterOffset,2)
cluster.display(points_3D,1)
cluster.display(points_3D,2)

# Random 2D geometry
points_2D=np.zeros((2,Number_points))
points_2D[0,:] = np.random.random(Number_points)
points_2D[1,:] = np.random.random(Number_points)

# Cluster 2D
cluster = Htool.Cluster(2)
cluster.set_minclustersize(minclustersize)
cluster.build(Number_points,points_2D,2)
cluster.display(points_2D,1)
cluster.display(points_2D,2)