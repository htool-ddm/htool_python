#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clustering/cluster_node.hpp"
#include "clustering/cluster_tree_builder.hpp"
#include "clustering/implementation/partitioning.hpp"
#include "clustering/interface/virtual_partitioning.hpp"
#include "clustering/utility.hpp"

#include "hmatrix/hmatrix.hpp"
#include "hmatrix/hmatrix_tree_builder.hpp"
#include "hmatrix/interfaces/virtual_dense_blocks_generator.hpp"
#include "hmatrix/interfaces/virtual_generator.hpp"
#include "hmatrix/interfaces/virtual_low_rank_generator.hpp"
#include "hmatrix/lrmat.hpp"

#include "local_operator/local_operator.hpp"
#include "local_operator/local_renumbering.hpp"
#include "local_operator/virtual_local_to_local_operator.hpp"

#ifdef HAVE_MPI
#    include "distributed_operator/distributed_operator.hpp"
#    include "distributed_operator/utility.hpp"

#    include "misc/wrapper_mpi.hpp"
#    include "solver/geneo/coarse_operator_builder.hpp"
#    include "solver/geneo/coarse_space_dense_builder.hpp"
#    include "solver/interfaces/virtual_coarse_operator_builder.hpp"
#    include "solver/interfaces/virtual_coarse_space_builder.hpp"
#    include "solver/solver.hpp"
#    include "solver/utility.hpp"
#endif

#include "matplotlib/cluster.hpp"
#include "matplotlib/hmatrix.hpp"
#include "misc/logger.hpp"
#include "misc/testing.hpp"

PYBIND11_MODULE(Htool, m) {
    // Delegate logging to python logging module
    htool::Logger::get_instance().set_current_writer(std::make_shared<PythonLoggerWriter>());
    m.def("test_logger", &test_logger);

#ifdef HAVE_MPI
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API."); // LCOV_EXCL_LINE
    }
#endif

    declare_cluster_node<double>(m, "Cluster");
    declare_virtual_partitioning<double>(m, "");
    declare_partitioning<double, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>(m, "PCARegular");
    declare_partitioning<double, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>(m, "PCAGeometric");
    declare_partitioning<double, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>(m, "BoundingBoxRegular");
    declare_partitioning<double, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>(m, "BoundingBoxGeometric");
    declare_cluster_builder<double>(m, "ClusterTreeBuilder");
    declare_cluster_utility<double>(m);

    declare_virtual_generator<double>(m, "VirtualGenerator", "IGenerator");
    declare_LowRankMatrix<double>(m, "LowRankMatrix");
    declare_HMatrix<double, double>(m, "HMatrix");
    declare_custom_VirtualLowRankGenerator<double>(m, "VirtualLowRankGenerator");
    declare_custom_VirtualDenseBlocksGenerator<double>(m, "VirtualDenseBlocksGenerator");
    declare_hmatrix_builder<double, double>(m, "HMatrixTreeBuilder");

#ifdef HAVE_MPI
    declare_local_renumbering<double>(m, "LocalRenumbering");
    declare_global_to_local_operator<double>(m, "");
    declare_virtual_local_to_local_operator<double>(m, "");
    declare_distributed_operator<double>(m, "DistributedOperator");
    declare_distributed_operator_utility<double, double>(m);

    declare_virtual_coarse_space_builder<double>(m, "VirtualGeneoCoarseSpaceBuilder", "ICoarseSpaceBuilder");
    declare_virtual_coarse_operator_builder<double>(m, "", "ICoarseOperatorBuilder");
    declare_virtual_geneo_coarse_space_dense_builder<double>(m, "VirtualGeneoCoarseSpaceDenseBuilder");
    declare_geneo_coarse_space_dense_builder<double>(m, "GeneoCoarseSpaceDenseBuilder");
    declare_geneo_coarse_operator_builder<double>(m, "GeneoCoarseOperatorBuilder");
    declare_DDM<double>(m, "Solver");

    declare_solver_utility<double, double>(m);
#endif

    declare_matplotlib_cluster<double>(m);
    declare_matplotlib_hmatrix<double, double>(m);

    declare_virtual_partitioning<std::complex<double>>(m, "Complex");
    declare_LowRankMatrix<std::complex<double>>(m, "ComplexLowRankMatrix");
    declare_HMatrix<std::complex<double>, double>(m, "ComplexHMatrix");
    declare_virtual_generator<std::complex<double>>(m, "ComplexVirtualGenerator", "IComplexGenerator");
    declare_custom_VirtualLowRankGenerator<std::complex<double>>(m, "VirtualComplexLowRankGenerator");
    declare_custom_VirtualDenseBlocksGenerator<std::complex<double>>(m, "ComplexVirtualDenseBlocksGenerator");
    declare_hmatrix_builder<std::complex<double>, double>(m, "ComplexHMatrixTreeBuilder");

#ifdef HAVE_MPI
    declare_global_to_local_operator<std::complex<double>>(m, "Complex");
    declare_virtual_local_to_local_operator<std::complex<double>>(m, "Complex");
    declare_distributed_operator<std::complex<double>>(m, "ComplexDistributedOperator");
    declare_distributed_operator_utility<std::complex<double>, double>(m, "Complex");

    declare_DDM<std::complex<double>>(m, "ComplexSolver");
    declare_virtual_coarse_space_builder<std::complex<double>>(m, "ComplexVirtualGeneoCoarseSpaceBuilder", "IComplexCoarseSpaceBuilder");
    declare_virtual_coarse_operator_builder<std::complex<double>>(m, "", "IComplexCoarseOperatorBuilder");
    declare_geneo_coarse_operator_builder<std::complex<double>>(m, "ComplexGeneoCoarseOperatorBuilder");
    declare_geneo_coarse_space_dense_builder<std::complex<double>>(m, "ComplexGeneoCoarseSpaceDenseBuilder");
    declare_virtual_geneo_coarse_space_dense_builder<std::complex<double>>(m, "VirtualComplexGeneoCoarseSpaceDenseBuilder");

    declare_solver_utility<std::complex<double>, double>(m, "Complex");
#endif
}
