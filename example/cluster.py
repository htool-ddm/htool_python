import Htool
import numpy as np

# Random geometry
NbRows = 500
NbCols = 250
np.random.seed(0)
points_target=np.zeros((NbRows,3))
points_target[:,0] = np.random.random(NbRows)
points_target[:,1] = np.random.random(NbRows)
points_target[:,2] = 1

r = np.zeros(NbRows,dtype=float)
g = np.zeros(NbRows,dtype=float)+1
tab = np.arange(NbRows,dtype=int)


# Cluster
cluster = Htool.Cluster()
cluster.build(points_target, r, tab, g,2)
cluster.display(points_target,2)