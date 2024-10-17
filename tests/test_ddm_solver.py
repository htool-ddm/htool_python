import copy

import Htool
import mpi4py
import numpy as np
import pytest
from conftest import GeneratorFromMatrix
from scipy.linalg import eig, eigh


class CustomGeneoCoarseSpaceDenseBuilder(
    Htool.VirtualComplexGeneoCoarseSpaceDenseBuilder
):
    def compute_coarse_space(self, Ai, Bi):
        coarse_space = None
        if self.symmetry == "S" or self.symmetry == "H":
            if self.geneo_threshold > 0:
                [w, v] = eigh(Ai, Bi, driver="gv")
            else:
                [w, v] = eigh(Ai, Bi, driver="gv")
        else:
            [w, v] = eig(Ai, Bi)

        idx = w.argsort()[::-1]

        if self.geneo_threshold > 0:
            nb_eig = (w > self.geneo_threshold).sum()
            coarse_space = v[:, idx[0:nb_eig]]
        else:
            coarse_space = v[:, idx[0 : self.geneo_nu]]
        self.set_coarse_space(coarse_space)


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("tol", [1e-6])
@pytest.mark.parametrize(
    "mu,symmetry,ddm_builder,hpddm_schwarz_method,hpddm_schwarz_coarse_correction,geneo_type",
    [
        (1, "N", "BlockJacobi", "none", "none", "none"),
        (1, "N", "BlockJacobi", "asm", "none", "none"),
        (1, "N", "BlockJacobi", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrix", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrix", "ras", "none", "none"),
        (10, "N", "BlockJacobi", "none", "none", "none"),
        (10, "N", "BlockJacobi", "asm", "none", "none"),
        (10, "N", "BlockJacobi", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrix", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrix", "ras", "none", "none"),
        (1, "S", "BlockJacobi", "none", "none", "none"),
        (1, "S", "BlockJacobi", "asm", "none", "none"),
        (1, "S", "BlockJacobi", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrix", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrix", "ras", "none", "none"),
        (10, "S", "BlockJacobi", "none", "none", "none"),
        (10, "S", "BlockJacobi", "asm", "none", "none"),
        (10, "S", "BlockJacobi", "ras", "none", "none"),
        (10, "S", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (10, "S", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "asm", "additive", "geneo_nu"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "ras", "additive", "geneo_nu"),
        (10, "S", "DDMWithHMatrixPlusOverlap", "asm", "additive", "geneo_nu"),
        (10, "S", "DDMWithHMatrixPlusOverlap", "ras", "additive", "geneo_nu"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "asm", "additive", "geneo_threshold"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "ras", "additive", "geneo_threshold"),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "geneo_threshold",
        ),
        (1, "S", "DDMWithHMatrixPlusOverlap", "asm", "additive", "custom_geneo_nu"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "ras", "additive", "custom_geneo_nu"),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_geneo_nu",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_geneo_nu",
        ),
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_geneo_threshold",
        ),
        (10, "S", "DDMWithHMatrix", "asm", "none", "none"),
        (10, "S", "DDMWithHMatrix", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrix", "asm", "additive", "geneo_nu"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "geneo_nu"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "geneo_nu"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "geneo_nu"),
        (1, "S", "DDMWithHMatrix", "asm", "additive", "geneo_threshold"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "geneo_threshold"),
        (1, "S", "DDMWithHMatrix", "asm", "additive", "custom_geneo_nu"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "custom_geneo_nu"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "custom_geneo_nu"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "custom_geneo_nu"),
        (1, "S", "DDMWithHMatrix", "asm", "additive", "custom_geneo_threshold"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "custom_geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "custom_geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "custom_geneo_threshold"),
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
    geneo_type,
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

    generator = GeneratorFromMatrix(A)
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

    solver = None
    default_solver_builder = None
    if ddm_builder == "BlockJacobi":
        block_diagonal_hmatrix = copy.deepcopy(
            default_approximation.block_diagonal_hmatrix
        )
        default_solver_builder = Htool.ComplexDDMSolverBuilder(
            default_approximation.distributed_operator,
            block_diagonal_hmatrix,
        )

    elif ddm_builder == "DDMWithHMatrixPlusOverlap":
        block_diagonal_hmatrix = copy.deepcopy(
            default_approximation.block_diagonal_hmatrix
        )
        default_solver_builder = Htool.ComplexDDMSolverBuilder(
            default_approximation.distributed_operator,
            block_diagonal_hmatrix,
            generator,
            ovr_subdomain_to_global,
            cluster_to_ovr_subdomain,
            neighbors,
            intersections,
        )

    elif ddm_builder == "DDMWithHMatrix":
        default_solver_builder = Htool.ComplexDDMSolverBuilder(
            default_approximation.distributed_operator,
            ovr_subdomain_to_global,
            cluster_to_ovr_subdomain,
            neighbors,
            intersections,
            generator,
            geometry,
            Htool.ClusterBuilder(),
            Htool.ComplexHMatrixBuilder(epsilon, eta * 1.0, symmetry, UPLO),
        )

    solver = default_solver_builder.solver
    distributed_operator = default_approximation.distributed_operator
    local_size_wo_overlap = len(cluster_to_ovr_subdomain)
    local_size_with_overlap = len(ovr_subdomain_to_global)

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
        geneo_space_operator_builder = None
        if geneo_type == "geneo_nu":
            geneo_space_operator_builder = Htool.ComplexGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "geneo_threshold":
            geneo_space_operator_builder = Htool.ComplexGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_threshold=100,
            )
        elif geneo_type == "custom_geneo_nu":
            geneo_space_operator_builder = CustomGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix.to_dense(),
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "custom_geneo_threshold":
            geneo_space_operator_builder = CustomGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix.to_dense(),
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_threshold=100,
            )
        geneo_coarse_operator_builder = Htool.ComplexGeneoCoarseOperatorBuilder(
            distributed_operator
        )
        solver.build_coarse_space(
            geneo_space_operator_builder, geneo_coarse_operator_builder
        )

        solver.set_hpddm_args(
            "-hpddm_schwarz_coarse_correction " + hpddm_schwarz_coarse_correction
        )

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

    if mpi4py.MPI.COMM_WORLD.rank == 0:
        print(solver.get_information())
    assert convergence_error < tol
    assert solution_error < epsilon * 10
