import Htool
import numpy as np
from mpi4py import MPI
import math
import struct
import os
import mpi4py
import pytest
import pathlib
from conftest import GeneratorFromMatrix


@pytest.mark.parametrize("epsilon", [1e-3, 1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("tol", [1e-10])
@pytest.mark.parametrize(
    "mu,symmetry,setup_solver_dependencies,hpddm_schwarz_method,hpddm_schwarz_coarse_correction",
    [
        (1, "S", True, "none", "none"),
        (1, "S", True, "asm", "none"),
        (1, "S", True, "ras", "none"),
        (10, "S", True, "none", "none"),
        (10, "S", True, "asm", "none"),
        (10, "S", True, "ras", "none"),
        (1, "N", True, "none", "none"),
        (1, "N", True, "asm", "none"),
        (1, "N", True, "ras", "none"),
        (10, "N", True, "none", "none"),
        (10, "N", True, "asm", "none"),
        (10, "N", True, "ras", "none"),
        # (1, "S", False, "asm", "additive"),
        # (1, "S", False, "ras", "additive"),
        # (10, "S", False, "asm", "additive"),
        # (10, "S", False, "ras", "additive"),
    ],
    indirect=["setup_solver_dependencies"],
)
def test_ddm_solver(
    setup_solver_dependencies,
    epsilon,
    mu,
    tol,
    hpddm_schwarz_method,
    hpddm_schwarz_coarse_correction,
):
    (
        solver,
        x_ref,
        f,
        distributed_operator,
        local_neumann_matrix,
    ) = setup_solver_dependencies

    x = np.zeros(len(f), dtype="complex128", order="F")
    if mu > 1:
        x = np.zeros((len(f), mu), dtype="complex128", order="F")

    solver.set_hpddm_args(
        "-hpddm_schwarz_method "
        + hpddm_schwarz_method
        + " -hpddm_max_it 200 -hpddm_gmres_restart 200 -hpddm_tol "
        + str(tol)
    )

    if hpddm_schwarz_coarse_correction != "none" and mpi4py.MPI.COMM_WORLD.size > 1:
        solver.set_hpddm_args(
            "-hpddm_schwarz_coarse_correction " + hpddm_schwarz_coarse_correction
        )
        # print(local_neumann_matrix, local_neumann_matrix.shape)
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
    assert convergence_error < tol
    assert solution_error < epsilon * 10
    # x.fill(0)

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
