import math
import mpi4py
import numpy as np


def create_random_points_in_disk(nb_points: int):
    u = np.random.rand(nb_points)
    theta = 2 * np.pi * np.random.rand(nb_points)

    x = np.sqrt(u) * np.cos(theta)
    y = np.sqrt(u) * np.sin(theta)
    return np.array([x, y])


def create_random_points_in_sphere(nb_points: int):
    u = np.random.rand(nb_points)
    theta = 2 * np.pi * np.random.rand(nb_points)
    phi = np.arccos(2 * np.random.rand(nb_points) - 1)

    r = np.cbrt(u)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def create_random_geometries(dimension: int, nb_rows: int, nb_cols: int):
    np.random.seed(0)

    if dimension == 3:
        target_points = create_random_points_in_sphere(nb_rows)
        source_points = create_random_points_in_sphere(nb_cols)

    if dimension == 2:
        target_points = create_random_points_in_disk(nb_rows)
        source_points = create_random_points_in_disk(nb_cols)

    source_points[0, :] += 2
    return [target_points, source_points]


def create_partitionned_geometries(
    dimension: int, nb_rows: int, nb_cols: int, sizeWorld: int
):
    np.random.seed(0)
    target_local_size = int(nb_rows / sizeWorld)
    target_partition = np.zeros((2, sizeWorld), dtype=int)

    if dimension == 2:
        target_points = np.array([[], []])
        for i in range(0, sizeWorld - 1):
            target_partition[0, i] = i * target_local_size
            target_partition[1, i] = target_local_size
            new_target_points = create_random_points_in_disk(target_partition[1, i])
            new_target_points[0, :] += 3 * i
            target_points = np.concatenate((target_points, new_target_points), axis=1)

        target_partition[0, sizeWorld - 1] = (sizeWorld - 1) * target_local_size
        target_partition[1, sizeWorld - 1] = (
            nb_rows - (sizeWorld - 1) * target_local_size
        )
        new_target_points = create_random_points_in_disk(
            target_partition[1, sizeWorld - 1]
        )
        new_target_points[0, :] += 3 * (sizeWorld - 1)
        target_points = np.concatenate((target_points, new_target_points), axis=1)
        source_points = create_random_points_in_disk(nb_cols)
        source_points[0, :] += 3 * (sizeWorld - 1) / 2
        source_points[1, :] += 3

    if dimension == 3:
        target_points = np.array([[], [], []])
        for i in range(0, sizeWorld - 1):
            target_partition[0, i] = i * target_local_size
            target_partition[1, i] = target_local_size
            new_target_points = create_random_points_in_sphere(target_partition[1, i])
            new_target_points[0, :] += 3 * i
            target_points = np.concatenate((target_points, new_target_points), axis=1)

        target_partition[0, sizeWorld - 1] = (sizeWorld - 1) * target_local_size
        target_partition[1, sizeWorld - 1] = (
            nb_rows - (sizeWorld - 1) * target_local_size
        )
        new_target_points = create_random_points_in_sphere(
            target_partition[1, sizeWorld - 1]
        )
        new_target_points[0, :] += 3 * (sizeWorld - 1)
        target_points = np.concatenate((target_points, new_target_points), axis=1)
        source_points = create_random_points_in_sphere(nb_cols)
        source_points[0, :] += 3 * (sizeWorld - 1) / 2
        source_points[1, :] += 3

    return [target_points, source_points, target_partition]
