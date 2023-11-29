import math
import os
import pathlib
import struct

import matplotlib.pyplot as plt
import mpi4py
import numpy as np
import pytest
from conftest import GeneratorFromMatrix, LocalGeneratorFromMatrix
from mpi4py import MPI

import Htool


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("tol", [1e-6])
@pytest.mark.parametrize(
    "mu,symmetry,ddm_builder,hpddm_schwarz_method,hpddm_schwarz_coarse_correction",
    [
        (1, "N", "SolverBuilder", "none", "none"),
        (1, "N", "SolverBuilder", "asm", "none"),
        (1, "N", "SolverBuilder", "ras", "none"),
        (1, "N", "DDMSolverBuilderAddingOverlap", "asm", "none"),
        (1, "N", "DDMSolverBuilderAddingOverlap", "ras", "none"),
        (1, "N", "DDMSolverBuilder", "asm", "none"),
        (1, "N", "DDMSolverBuilder", "ras", "none"),
        (10, "N", "SolverBuilder", "none", "none"),
        (10, "N", "SolverBuilder", "asm", "none"),
        (10, "N", "SolverBuilder", "ras", "none"),
        (10, "N", "DDMSolverBuilderAddingOverlap", "asm", "none"),
        (10, "N", "DDMSolverBuilderAddingOverlap", "ras", "none"),
        (10, "N", "DDMSolverBuilder", "asm", "none"),
        (10, "N", "DDMSolverBuilder", "ras", "none"),
        (1, "S", "SolverBuilder", "none", "none"),
        (1, "S", "SolverBuilder", "asm", "none"),
        (1, "S", "SolverBuilder", "ras", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "none"),
        (1, "S", "DDMSolverBuilder", "asm", "none"),
        (1, "S", "DDMSolverBuilder", "ras", "none"),
        (10, "S", "SolverBuilder", "none", "none"),
        (10, "S", "SolverBuilder", "asm", "none"),
        (10, "S", "SolverBuilder", "ras", "none"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "asm", "none"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "ras", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive"),
        (10, "S", "DDMSolverBuilder", "asm", "none"),
        (10, "S", "DDMSolverBuilder", "ras", "none"),
        (1, "S", "DDMSolverBuilder", "asm", "additive"),
        (1, "S", "DDMSolverBuilder", "ras", "additive"),
        (10, "S", "DDMSolverBuilder", "asm", "additive"),
        (10, "S", "DDMSolverBuilder", "ras", "additive"),
    ],
    # indirect=["setup_solver_dependencies"],
)
def test_ddm_solver(
    load_data_solver,
    epsilon,
    eta,
    mu,
    ddm_builder,
    symmetry,
    tol,
    hpddm_schwarz_method,
    hpddm_schwarz_coarse_correction,
):
    # (
    #     solver,
    #     x_ref,
    #     f,
    #     distributed_operator,
    #     local_neumann_matrix,
    # ) = setup_solver_dependencies

    # Setup
    [
        A,
        x_ref,
        f,
        geometry,
        cluster,
        neighbors,
        intersections,
        symmetry,
        UPLO,
        cluster_to_ovr_subdomain,
        ovr_subdomain_to_global,
        local_neumann_matrix,
    ] = load_data_solver

    generator = GeneratorFromMatrix(cluster.get_permutation(), A)
    default_approximation = Htool.ComplexDefaultApproximationBuilder(
        generator,
        cluster,
        cluster,
        epsilon,
        eta,
        symmetry,
        UPLO,
        mpi4py.MPI.COMM_WORLD,
    )
    # print("Geometry", geometry)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection="3d")
    # ax.scatter(geometry[0, :], geometry[1, :], geometry[2, :], marker="o")
    # plt.show()
    solver = None
    if ddm_builder == "SolverBuilder":
        default_solver_builder = Htool.ComplexDefaultSolverBuilder(
            default_approximation.distributed_operator,
            default_approximation.block_diagonal_hmatrix,
        )
        solver = default_solver_builder.solver
    elif ddm_builder == "DDMSolverBuilderAddingOverlap":
        default_solver_builder = Htool.ComplexDefaultDDMSolverBuilderAddingOverlap(
            default_approximation.distributed_operator,
            default_approximation.block_diagonal_hmatrix,
            generator,
            ovr_subdomain_to_global,
            cluster_to_ovr_subdomain,
            neighbors,
            intersections,
        )
        solver = default_solver_builder.solver
    elif ddm_builder == "DDMSolverBuilder":
        local_numbering_builder = Htool.LocalNumberingBuilder(
            ovr_subdomain_to_global,
            cluster_to_ovr_subdomain,
            intersections,
        )
        intersections = local_numbering_builder.intersections
        local_to_global_numbering = local_numbering_builder.local_to_global_numbering
        local_geometry = geometry[:, local_to_global_numbering]

        local_cluster_builder = Htool.ClusterBuilder()

        local_cluster: Htool.Cluster = local_cluster_builder.create_cluster_tree(
            local_geometry, 2, 2
        )

        local_hmatrix_builder = Htool.ComplexHMatrixBuilder(
            local_cluster,
            local_cluster,
            epsilon,
            eta,
            symmetry,
            UPLO,
            -1,
            -1,
        )
        local_generator = LocalGeneratorFromMatrix(
            local_cluster.get_permutation(), local_to_global_numbering, A
        )
        local_hmatrix = local_hmatrix_builder.build(local_generator)
        default_solver_builder = Htool.ComplexDefaultDDMSolverBuilder(
            default_approximation.distributed_operator,
            local_hmatrix,
            neighbors,
            intersections,
        )
        solver = default_solver_builder.solver

    distributed_operator = default_approximation.distributed_operator

    # Solver
    x = np.zeros(len(f), dtype="complex128", order="F")
    if mu > 1:
        x = np.zeros((len(f), mu), dtype="complex128", order="F")
    iterative_solver = "gmres"
    restart = "" if (symmetry == "S") else " -hpddm_gmres_restart 200 "
    hpddm_args = (
        "-hpddm_krylov_method "
        + iterative_solver
        + restart
        + " -hpddm_schwarz_method "
        + hpddm_schwarz_method
        + " -hpddm_max_it 200 -hpddm_variant right -hpddm_tol "
        + str(tol)
    )

    # if mpi4py.MPI.COMM_WORLD.rank == 0:
    #     hpddm_args += " -hpddm_verbosity 100 "

    solver.set_hpddm_args(hpddm_args)

    if hpddm_schwarz_coarse_correction != "none" and mpi4py.MPI.COMM_WORLD.size > 1:
        solver.set_hpddm_args(
            "-hpddm_schwarz_coarse_correction " + hpddm_schwarz_coarse_correction
        )
        solver.build_coarse_space(local_neumann_matrix)

    if hpddm_schwarz_method == "asm" or hpddm_schwarz_method == "ras":
        solver.facto_one_level()

    # No precond wo overlap
    solver.solve(
        x,
        f,
    )

    if mu == 1:
        convergence_error = np.linalg.norm(
            distributed_operator * x - f
        ) / np.linalg.norm(f)
        solution_error = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    elif mu > 1:
        convergence_error = np.linalg.norm(
            distributed_operator @ x - f
        ) / np.linalg.norm(f)
        solution_error = np.linalg.norm(x[:, 1] - x_ref) / np.linalg.norm(x_ref)
    # error = np.linalg.norm(distributed_operator * x - f)
    # if mpi4py.MPI.COMM_WORLD.rank == 0:
    #     print(
    #         iterative_solver,
    #         hpddm_schwarz_method,
    #         hpddm_schwarz_coarse_correction,
    #         epsilon,
    #         solver.get_information("Nb_it"),
    #         # error,
    #         # np.linalg.norm(f),
    #         # error / np.linalg.norm(f),
    #         # hpddm_args,
    #     )
    # print(
    #     np.linalg.norm(distributed_operator * x - f),
    #     np.linalg.norm(f),
    #     tol,
    #     mpi4py.MPI.COMM_WORLD.rank,
    # )
    if mpi4py.MPI.COMM_WORLD.rank == 0:
        print(solver.get_information())
    assert convergence_error < tol
    assert solution_error < epsilon * 10

    # # DDM one level ASM wo overlap
    # if rank == 0:
    #     print("ASM one level without overlap:")
    # comm.Barrier()
    # block_jacobi.set_hpddm_args("-hpddm_schwarz_method asm")
    # block_jacobi.facto_one_level()
    # block_jacobi.solve(x, f)
    # block_jacobi.print_infos()
    # if mu == 1:
    #     error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    # elif mu > 1:
    #     error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    # if rank == 0:
    #     print(error)
    # assert error < tol
    # x.fill(0)

    # # DDM one level ASM wo overlap
    # if rank == 0:
    #     print("RAS one level without overlap:")
    # comm.Barrier()
    # block_jacobi.set_hpddm_args("-hpddm_schwarz_method ras")
    # block_jacobi.solve(x, f)
    # block_jacobi.print_infos()
    # if mu == 1:
    #     error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    # elif mu > 1:
    #     error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    # if rank == 0:
    #     print(error)
    # assert error < tol
    # x.fill(0)

    # # Check infos
    # if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    #     assert mpi4py.MPI.COMM_WORLD.Get_size() == int(
    #         block_jacobi.get_infos("Nb_subdomains")
    #     )

    # # No precond with overlap
    # if rank == 0:
    #     print("No precond with overlap:")
    # DDM_solver.solve(
    #     x,
    #     f,
    #     hpddm_args="-hpddm_schwarz_method none -hpddm_max_it 200 -hpddm_tol "
    #     + str(tol),
    # )
    # DDM_solver.print_infos()
    # if mu == 1:
    #     error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    # elif mu > 1:
    #     error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    # if rank == 0:
    #     print(error)
    # assert error < tol
    # x.fill(0)

    # # DDM one level ASM with overlap
    # if rank == 0:
    #     print("ASM one level with overlap:")
    # comm.Barrier()
    # DDM_solver.set_hpddm_args("-hpddm_schwarz_method asm")
    # if Symmetry == "S" and size > 1:
    #     # Matrix
    #     with open(
    #         os.path.join(
    #             os.path.dirname(__file__)
    #             + "/../lib/htool/data/data_test/"
    #             + folder
    #             + "/Ki_"
    #             + str(size)
    #             + "_"
    #             + str(rank)
    #             + ".bin"
    #         ),
    #         "rb",
    #     ) as input:
    #         data = input.read()
    #         (m, n) = struct.unpack("@II", data[:8])
    #         # print(m,n)
    #         Ki = np.frombuffer(data[8:], dtype=np.dtype("complex128"))
    #         Ki = np.transpose(Ki.reshape((m, n)))
    #     DDM_solver.build_coarse_space(Ki)
    # DDM_solver.facto_one_level()
    # DDM_solver.solve(x, f)
    # DDM_solver.print_infos()
    # if mu == 1:
    #     error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    # elif mu > 1:
    #     error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    # if rank == 0:
    #     print(error)
    # assert error < tol
    # x.fill(0)

    # # DDM one level RAS with overlap
    # if rank == 0:
    #     print("RAS one level with overlap:")
    # comm.Barrier()
    # DDM_solver.set_hpddm_args("-hpddm_schwarz_method ras")
    # DDM_solver.solve(x, f)
    # DDM_solver.print_infos()
    # if mu == 1:
    #     error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    # elif mu > 1:
    #     error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    # if rank == 0:
    #     print(error)
    # assert error < tol
    # x.fill(0)

    # # DDM two level ASM with overlap
    # if Symmetry == "S" and size > 1:
    #     if rank == 0:
    #         print("ASM two level with overlap:")
    #     comm.Barrier()
    #     DDM_solver.set_hpddm_args(
    #         "-hpddm_schwarz_method asm -hpddm_schwarz_coarse_correction additive"
    #     )
    #     DDM_solver.solve(x, f)
    #     DDM_solver.print_infos()
    #     if mu == 1:
    #         error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    #     elif mu > 1:
    #         error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    #     if rank == 0:
    #         print(error)
    #     assert error < tol
    #     x.fill(0)

    #     if rank == 0:
    #         print("RAS two level with overlap:")
    #     comm.Barrier()
    #     DDM_solver.set_hpddm_args(
    #         "-hpddm_schwarz_method asm -hpddm_schwarz_coarse_correction additive"
    #     )
    #     DDM_solver.solve(x, f)
    #     DDM_solver.print_infos()
    #     if mu == 1:
    #         error = np.linalg.norm(hmat * x - f) / np.linalg.norm(f)
    #     elif mu > 1:
    #         error = np.linalg.norm(hmat @ x - f) / np.linalg.norm(f)
    #     if rank == 0:
    #         print(error)
    #     assert error < tol
    #     x.fill(0)

    # # Check infos
    # if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    #     assert mpi4py.MPI.COMM_WORLD.Get_size() == int(
    #         DDM_solver.get_infos("Nb_subdomains")
    #     )
