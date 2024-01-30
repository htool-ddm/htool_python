import Htool
import mpi4py
import numpy as np
import pytest
from conftest import GeneratorFromMatrix, LocalGeneratorFromMatrix
from scipy.linalg import eig, eigh


class CustomGeneoCoarseSpaceDenseBuilder(
    Htool.VirtualComplexGeneoCoarseSpaceDenseBuilder
):
    def compute_coarse_space(self, Ai, Bi):
        coarse_space = None

        if self.symmetry == "S" or self.symmetry == "H":
            if self.geneo_threshold > 0:
                [_, coarse_space] = eigh(
                    Ai, Bi, subset_by_value=[self.geneo_threshold, np.inf]
                )
            else:
                n = Ai.shape[0]
                [_, coarse_space] = eigh(
                    Ai, Bi, subset_by_index=[n - self.geneo_nu, n - 1]
                )
        else:
            [w, v] = eig(Ai, Bi)
            if self.geneo_threshold > 0:
                nb_eig = (w > self.geneo_threshold).sum()
                coarse_space = v[:, 0:nb_eig]
            else:
                coarse_space = v[:, 0 : self.geneo_nu]

        self.set_coarse_space(coarse_space)


@pytest.mark.parametrize("epsilon", [1e-6])
@pytest.mark.parametrize("eta", [10])
@pytest.mark.parametrize("tol", [1e-6])
@pytest.mark.parametrize(
    "mu,symmetry,ddm_builder,hpddm_schwarz_method,hpddm_schwarz_coarse_correction,geneo_type",
    [
        (1, "N", "SolverBuilder", "none", "none", "none"),
        (1, "N", "SolverBuilder", "asm", "none", "none"),
        (1, "N", "SolverBuilder", "ras", "none", "none"),
        (1, "N", "DDMSolverBuilderAddingOverlap", "asm", "none", "none"),
        (1, "N", "DDMSolverBuilderAddingOverlap", "ras", "none", "none"),
        (1, "N", "DDMSolverBuilder", "asm", "none", "none"),
        (1, "N", "DDMSolverBuilder", "ras", "none", "none"),
        (10, "N", "SolverBuilder", "none", "none", "none"),
        (10, "N", "SolverBuilder", "asm", "none", "none"),
        (10, "N", "SolverBuilder", "ras", "none", "none"),
        (10, "N", "DDMSolverBuilderAddingOverlap", "asm", "none", "none"),
        (10, "N", "DDMSolverBuilderAddingOverlap", "ras", "none", "none"),
        (10, "N", "DDMSolverBuilder", "asm", "none", "none"),
        (10, "N", "DDMSolverBuilder", "ras", "none", "none"),
        (1, "S", "SolverBuilder", "none", "none", "none"),
        (1, "S", "SolverBuilder", "asm", "none", "none"),
        (1, "S", "SolverBuilder", "ras", "none", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "none", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "none", "none"),
        (1, "S", "DDMSolverBuilder", "asm", "none", "none"),
        (1, "S", "DDMSolverBuilder", "ras", "none", "none"),
        (10, "S", "SolverBuilder", "none", "none", "none"),
        (10, "S", "SolverBuilder", "asm", "none", "none"),
        (10, "S", "SolverBuilder", "ras", "none", "none"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "asm", "none", "none"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "ras", "none", "none"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive", "geneo_nu"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive", "geneo_nu"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive", "geneo_nu"),
        (10, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive", "geneo_nu"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive", "geneo_threshold"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive", "geneo_threshold"),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "asm",
            "additive",
            "geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "ras",
            "additive",
            "geneo_threshold",
        ),
        (1, "S", "DDMSolverBuilderAddingOverlap", "asm", "additive", "custom_geneo_nu"),
        (1, "S", "DDMSolverBuilderAddingOverlap", "ras", "additive", "custom_geneo_nu"),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "asm",
            "additive",
            "custom_geneo_nu",
        ),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "ras",
            "additive",
            "custom_geneo_nu",
        ),
        (
            1,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "asm",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            1,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "ras",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "asm",
            "additive",
            "custom_geneo_threshold",
        ),
        (
            10,
            "S",
            "DDMSolverBuilderAddingOverlap",
            "ras",
            "additive",
            "custom_geneo_threshold",
        ),
        (10, "S", "DDMSolverBuilder", "asm", "none", "none"),
        (10, "S", "DDMSolverBuilder", "ras", "none", "none"),
        (1, "S", "DDMSolverBuilder", "asm", "additive", "geneo_nu"),
        (1, "S", "DDMSolverBuilder", "ras", "additive", "geneo_nu"),
        (10, "S", "DDMSolverBuilder", "asm", "additive", "geneo_nu"),
        (10, "S", "DDMSolverBuilder", "ras", "additive", "geneo_nu"),
        (1, "S", "DDMSolverBuilder", "asm", "additive", "geneo_threshold"),
        (1, "S", "DDMSolverBuilder", "ras", "additive", "geneo_threshold"),
        (10, "S", "DDMSolverBuilder", "asm", "additive", "geneo_threshold"),
        (10, "S", "DDMSolverBuilder", "ras", "additive", "geneo_threshold"),
        (1, "S", "DDMSolverBuilder", "asm", "additive", "custom_geneo_nu"),
        (1, "S", "DDMSolverBuilder", "ras", "additive", "custom_geneo_nu"),
        (10, "S", "DDMSolverBuilder", "asm", "additive", "custom_geneo_nu"),
        (10, "S", "DDMSolverBuilder", "ras", "additive", "custom_geneo_nu"),
        (1, "S", "DDMSolverBuilder", "asm", "additive", "custom_geneo_threshold"),
        (1, "S", "DDMSolverBuilder", "ras", "additive", "custom_geneo_threshold"),
        (10, "S", "DDMSolverBuilder", "asm", "additive", "custom_geneo_threshold"),
        (10, "S", "DDMSolverBuilder", "ras", "additive", "custom_geneo_threshold"),
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

    solver = None
    default_solver_builder = None
    if ddm_builder == "SolverBuilder":
        default_solver_builder = Htool.ComplexDefaultSolverBuilder(
            default_approximation.distributed_operator,
            default_approximation.block_diagonal_hmatrix,
        )

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

        geneo_space_operator_builder = None
        if geneo_type == "geneo_nu":
            geneo_space_operator_builder = Htool.ComplexGeneoCoarseSpaceDenseBuilder(
                cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank).get_size(),
                default_solver_builder.block_diagonal_dense_matrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "geneo_threshold":
            geneo_space_operator_builder = Htool.ComplexGeneoCoarseSpaceDenseBuilder(
                cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank).get_size(),
                default_solver_builder.block_diagonal_dense_matrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_threshold=100,
            )
        elif geneo_type == "custom_geneo_nu":
            geneo_space_operator_builder = CustomGeneoCoarseSpaceDenseBuilder(
                cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank).get_size(),
                default_solver_builder.block_diagonal_dense_matrix,
                local_neumann_matrix,
                symmetry,
                UPLO,
                geneo_nu=2,
            )
        elif geneo_type == "custom_geneo_threshold":
            geneo_space_operator_builder = Htool.ComplexGeneoCoarseSpaceDenseBuilder(
                cluster.get_cluster_on_partition(mpi4py.MPI.COMM_WORLD.rank).get_size(),
                default_solver_builder.block_diagonal_dense_matrix,
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
