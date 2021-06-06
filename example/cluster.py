import Htool
import numpy as np

# Parameters
minclustersize = 5

# Random 3D geometry
Number_points = 500
np.random.seed(0)
points_3D=np.zeros((3,Number_points))
points_3D[0,:] = np.random.random(Number_points)
points_3D[1,:] = np.random.random(Number_points)
points_3D[2,:] = np.random.random(Number_points)

# Cluster 3D
cluster = Htool.Cluster(3)
cluster.set_minclustersize(minclustersize)
cluster.build(Number_points,points_3D,2)
cluster.display(points_3D,2)

# Random 2D geometry
points_2D=np.zeros((2,Number_points))
points_2D[0,:] = np.random.random(Number_points)
points_2D[1,:] = np.random.random(Number_points)

# Cluster 2D
cluster = Htool.Cluster(2)
cluster.set_minclustersize(minclustersize)
cluster.build(Number_points,points_2D,2)
cluster.display(points_2D,2)