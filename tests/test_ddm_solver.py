import copy
import logging

import mpi4py
import numpy as np
import pytest
from conftest import ComplexGeneratorFromMatrix, GeneratorFromMatrix
from scipy.linalg import eig, eigh, ldl, solve_triangular
from scipy.sparse.linalg import LinearOperator, eigsh

import Htool


class CustomDenseGeneoCoarseSpaceDenseBuilder(
    Htool.VirtualGeneoCoarseSpaceDenseBuilder
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
            coarse_space = v[:, idx[0:nb_eig]].real
        else:
            coarse_space = v[:, idx[0 : self.geneo_nu]].real
        self.set_coarse_space(coarse_space)


class CustomGeneoCoarseSpaceBuilder(Htool.VirtualGeneoCoarseSpaceBuilder):
    class DirichletHMatrix(LinearOperator):
        def __init__(self, hmatrix_callback, size, dtype):
            self.hmatrix_callback = hmatrix_callback
            self.shape = (size, size)
            self.dtype = dtype

        def _matvec(self, x):
            return self.hmatrix_callback(x)

    class DenseNeumannMatrix(LinearOperator):
        def __init__(self, Bi):
            self.L, self.D, self.perm = ldl(Bi)
            self.shape = Bi.shape
            self.dtype = Bi.dtype

        def _matvec(self, x):
            tmp = solve_triangular(self.L[self.perm, :], x[self.perm], lower=True)
            for index, coef in enumerate(self.D.diagonal()):
                #     if coef>1e-4:
                tmp[index] /= coef
            #     else :
            #         tmp[index]=0
            res = np.empty_like(x)
            res[self.perm] = solve_triangular(
                self.L[self.perm, :], tmp, lower=True, trans="T"
            )
            return res

    def __init__(
        self,
        size_wo_overlap,
        size_with_overlap,
        Ai,
        Bi,
        geneo_nu=2,
        geneo_threshold=None,
    ):
        if not geneo_threshold:
            Htool.VirtualGeneoCoarseSpaceBuilder.__init__(
                self,
                size_wo_overlap,
                size_with_overlap,
                Ai,
                geneo_nu=geneo_nu,
            )
        else:
            Htool.VirtualGeneoCoarseSpaceBuilder.__init__(
                self,
                size_wo_overlap,
                size_with_overlap,
                Ai,
                geneo_threshold=geneo_threshold,
            )
        self.local_size = size_with_overlap
        self.Bi = Bi
        self.Ai = Ai

    def compute_coarse_space(self, hmatrix_callback):
        self.coarse_space = None
        self.eigenvalues = None

        if self.geneo_threshold > 0:
            [w, v] = eigsh(
                self.Bi,
                10,
                M=self.DirichletHMatrix(
                    hmatrix_callback, self.local_size, np.dtype("float64")
                ),
                OPinv=self.DenseNeumannMatrix(self.Bi),
                return_eigenvectors=True,
                sigma=0,
            )
        else:
            [w, v] = eigsh(
                self.Bi,
                self.geneo_nu,
                M=self.DirichletHMatrix(
                    hmatrix_callback, self.local_size, np.dtype("float64")
                ),
                OPinv=self.DenseNeumannMatrix(self.Bi),
                return_eigenvectors=True,
                sigma=0,
            )
        idx = w.argsort()
        if self.geneo_threshold > 0:
            nb_eig = (w < 1.0 / self.geneo_threshold).sum()
            self.coarse_space = v[:, idx[0:nb_eig]].real
            self.eigenvalues = 1.0 / w[idx[:]]
        else:
            self.coarse_space = v[:, idx[0 : self.geneo_nu]]
            self.eigenvalues = 1.0 / w[idx[:]]

        self.set_coarse_space(self.coarse_space)


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("tol", [1e-6])
@pytest.mark.parametrize(
    "mu,symmetry,ddm_builder,hpddm_schwarz_method,hpddm_schwarz_coarse_correction,geneo_type",
    [
        (1, "N", "BlockJacobi", "none", "none", "none"),
        (1, "N", "BlockJacobi", "asm", "none", "none"),
        (1, "N", "BlockJacobi", "ras", "none", "none"),
        (1, "N", "BlockJacobiDense", "none", "none", "none"),
        (1, "N", "BlockJacobiDense", "asm", "none", "none"),
        (1, "N", "BlockJacobiDense", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrix", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrix", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrixDense", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrixDense", "ras", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlapDense", "asm", "none", "none"),
        (1, "N", "DDMWithHMatrixPlusOverlapDense", "ras", "none", "none"),
        (10, "N", "BlockJacobi", "none", "none", "none"),
        (10, "N", "BlockJacobi", "asm", "none", "none"),
        (10, "N", "BlockJacobi", "ras", "none", "none"),
        (10, "N", "BlockJacobiDense", "none", "none", "none"),
        (10, "N", "BlockJacobiDense", "asm", "none", "none"),
        (10, "N", "BlockJacobiDense", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrix", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrix", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlapDense", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrixPlusOverlapDense", "ras", "none", "none"),
        (10, "N", "DDMWithHMatrixDense", "asm", "none", "none"),
        (10, "N", "DDMWithHMatrixDense", "ras", "none", "none"),
        (1, "S", "BlockJacobi", "none", "none", "none"),
        (1, "S", "BlockJacobi", "asm", "none", "none"),
        (1, "S", "BlockJacobi", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlap", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrix", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrix", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlapDense", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrixPlusOverlapDense", "ras", "none", "none"),
        (1, "S", "DDMWithHMatrixDense", "asm", "none", "none"),
        (1, "S", "DDMWithHMatrixDense", "ras", "none", "none"),
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
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_dense_geneo_nu",
        ),
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_dense_geneo_nu",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_dense_geneo_nu",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_dense_geneo_nu",
        ),
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_dense_geneo_threshold",
        ),
        (
            1,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_dense_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "asm",
            "additive",
            "custom_dense_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMWithHMatrixPlusOverlap",
            "ras",
            "additive",
            "custom_dense_geneo_threshold",
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
        (1, "S", "DDMWithHMatrix", "asm", "additive", "custom_dense_geneo_nu"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "custom_dense_geneo_nu"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "custom_dense_geneo_nu"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "custom_dense_geneo_nu"),
        (1, "S", "DDMWithHMatrix", "asm", "additive", "custom_dense_geneo_threshold"),
        (1, "S", "DDMWithHMatrix", "ras", "additive", "custom_dense_geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "asm", "additive", "custom_dense_geneo_threshold"),
        (10, "S", "DDMWithHMatrix", "ras", "additive", "custom_dense_geneo_threshold"),
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
    logging.basicConfig(level=logging.INFO)

    if symmetry == "S":
        generator = GeneratorFromMatrix(A)
        default_approximation = Htool.DefaultApproximationBuilder(
            generator,
            cluster,
            cluster,
            Htool.HMatrixTreeBuilder(epsilon, eta, symmetry, UPLO),
            mpi4py.MPI.COMM_WORLD,
        )
    else:
        generator = ComplexGeneratorFromMatrix(A)
        default_approximation = Htool.ComplexDefaultApproximationBuilder(
            generator,
            cluster,
            cluster,
            Htool.ComplexHMatrixTreeBuilder(epsilon, eta, symmetry, UPLO),
            mpi4py.MPI.COMM_WORLD,
        )
    Htool.recompression(default_approximation.hmatrix)

    solver = None
    default_solver_builder = None
    if ddm_builder == "BlockJacobi":
        block_diagonal_hmatrix = copy.deepcopy(
            default_approximation.block_diagonal_hmatrix
        )
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverBuilder(
                default_approximation.distributed_operator,
                block_diagonal_hmatrix,
            )
        else:
            default_solver_builder = Htool.ComplexDDMSolverBuilder(
                default_approximation.distributed_operator,
                block_diagonal_hmatrix,
            )
    elif ddm_builder == "BlockJacobiDense":
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                default_approximation.block_diagonal_hmatrix,
            )
        else:
            default_solver_builder = Htool.ComplexDDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                default_approximation.block_diagonal_hmatrix,
            )
        local_hmatrix = default_solver_builder.get_local_hmatrix()

    elif ddm_builder == "DDMWithHMatrixPlusOverlap":
        block_diagonal_hmatrix = copy.deepcopy(
            default_approximation.block_diagonal_hmatrix
        )
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverBuilder(
                default_approximation.distributed_operator,
                block_diagonal_hmatrix,
                generator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
            )
        else:
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
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverBuilder(
                default_approximation.distributed_operator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
                generator,
                geometry,
                Htool.ClusterTreeBuilder(),
                Htool.HMatrixTreeBuilder(epsilon, eta * 1.0, symmetry, UPLO),
                radii=None,
                weights=None,
            )
        else:
            default_solver_builder = Htool.ComplexDDMSolverBuilder(
                default_approximation.distributed_operator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
                generator,
                geometry,
                Htool.ClusterTreeBuilder(),
                Htool.ComplexHMatrixTreeBuilder(epsilon, eta * 1.0, symmetry, UPLO),
                radii=None,
                weights=None,
            )
        local_hmatrix = default_solver_builder.get_local_hmatrix()
        Htool.recompression(local_hmatrix)

    elif ddm_builder == "DDMWithHMatrixPlusOverlapDense":
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                default_approximation.block_diagonal_hmatrix,
                generator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
            )
        else:
            default_solver_builder = Htool.ComplexDDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                default_approximation.block_diagonal_hmatrix,
                generator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
            )
        local_hmatrix = default_solver_builder.get_local_hmatrix()
    elif ddm_builder == "DDMWithHMatrixDense":
        if symmetry == "S":
            default_solver_builder = Htool.DDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
                generator,
                geometry,
                Htool.HMatrixTreeBuilder(epsilon, eta * 1.0, symmetry, UPLO),
            )
        else:
            default_solver_builder = Htool.ComplexDDMSolverWithDenseLocalSolver(
                default_approximation.distributed_operator,
                ovr_subdomain_to_global,
                cluster_to_ovr_subdomain,
                neighbors,
                intersections,
                generator,
                geometry,
                Htool.ComplexHMatrixTreeBuilder(epsilon, eta * 1.0, symmetry, UPLO),
            )
        local_hmatrix = default_solver_builder.get_local_hmatrix()

    solver = default_solver_builder.solver
    distributed_operator = default_approximation.distributed_operator
    local_size_wo_overlap = len(cluster_to_ovr_subdomain)
    local_size_with_overlap = len(ovr_subdomain_to_global)

    # Solver
    dtype = "complex128" if symmetry != "S" else "float64"
    x = np.zeros(len(f), dtype=dtype, order="F")
    if mu > 1:
        x = np.zeros((len(f), mu), dtype=dtype, order="F")
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
            geneo_space_operator_builder = Htool.GeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "geneo_threshold":
            geneo_space_operator_builder = Htool.GeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_threshold=100,
            )
        elif geneo_type == "custom_geneo_nu":
            geneo_space_operator_builder = CustomGeneoCoarseSpaceBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix
                if ddm_builder == "DDMWithHMatrixDensePlusOverlap"
                or ddm_builder == "DDMWithHMatrixPlusOverlap"
                else default_solver_builder.get_local_hmatrix(),
                local_neumann_matrix,
                geneo_nu=2,
            )
        elif geneo_type == "custom_geneo_threshold":
            geneo_space_operator_builder = CustomGeneoCoarseSpaceBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix,
                local_neumann_matrix,
                geneo_threshold=100,
            )
        elif geneo_type == "custom_dense_geneo_nu":
            geneo_space_operator_builder = CustomDenseGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix.to_dense(),
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "custom_dense_geneo_threshold":
            geneo_space_operator_builder = CustomDenseGeneoCoarseSpaceDenseBuilder(
                local_size_wo_overlap,
                local_size_with_overlap,
                default_approximation.block_diagonal_hmatrix.to_dense(),
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_threshold=100,
            )
        geneo_coarse_operator_builder = Htool.GeneoCoarseOperatorBuilder(
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
