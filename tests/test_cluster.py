import Htool
import matplotlib.pyplot as plt
import mpi4py
import pytest


@pytest.mark.parametrize(
    "dimension,nb_rows,nb_cols,symmetry,is_partition_given",
    [
        (2, 500, 500, "N", False),
        (3, 500, 500, "N", False),
        (2, 500, 500, "N", True),
        (3, 500, 500, "N", True),
    ],
)
def test_cluster(geometry, cluster):
    [target_points, _, _] = geometry
    [target_cluster, _] = cluster
    local_target_cluster = target_cluster.get_cluster_on_partition(
        mpi4py.MPI.COMM_WORLD.Get_rank()
    )

    total_size = 0
    for p in range(0, mpi4py.MPI.COMM_WORLD.Get_size()):
        total_size += target_cluster.get_cluster_on_partition(p).get_size()

    assert total_size == len(local_target_cluster.get_permutation())
    assert total_size == len(target_cluster.get_permutation())

    # Several ways to display information
    if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
        _, ax = plt.subplots(2, 2)
        Htool.plot(ax[0, 0], target_cluster, target_points, 1)
        Htool.plot(ax[0, 1], target_cluster, target_points, 2)
        Htool.plot(ax[1, 0], local_target_cluster, target_points, 1)
        Htool.plot(ax[1, 1], local_target_cluster, target_points, 2)
