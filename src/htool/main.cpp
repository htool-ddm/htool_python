#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "clustering/cluster_builder.hpp"
#include "clustering/cluster_node.hpp"
#include "clustering/implementation/direction_computation.hpp"
#include "clustering/implementation/splitting.hpp"
#include "clustering/interface/direction_computation.hpp"
#include "clustering/interface/splitting.hpp"
#include "clustering/utility.hpp"

#include "hmatrix/hmatrix.hpp"
#include "hmatrix/hmatrix_builder.hpp"
#include "hmatrix/interfaces/virtual_dense_blocks_generator.hpp"
#include "hmatrix/interfaces/virtual_generator.hpp"
#include "hmatrix/interfaces/virtual_low_rank_generator.hpp"

#include "local_operator/local_dense_operator.hpp"
#include "local_operator/local_hmatrix.hpp"
#include "local_operator/local_operator.hpp"
#include "local_operator/virtual_local_operator.hpp"

#include "distributed_operator/distributed_operator.hpp"
#include "distributed_operator/implementation/partition_from_cluster.hpp"
#include "distributed_operator/utility.hpp"

#include "solver/solver.hpp"
#include "solver/utility.hpp"

#include "matplotlib/cluster.hpp"
#include "matplotlib/hmatrix.hpp"
#include "misc/logger.hpp"
// #include "ddm_solver.hpp"
// #include "dense_blocks_generator.hpp"
// #include "hmatrix.hpp"
// #include "lrmat_generator.hpp"
// #include "matrix.hpp"

PYBIND11_MODULE(Htool, m) {
    // Delegate logging to python logging module
    Logger::get_instance().set_current_writer(std::make_shared<PythonLoggerWriter>());

    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API."); // LCOV_EXCL_LINE
    }

    declare_cluster_node<double>(m, "Cluster");
    declare_cluster_builder<double>(m, "ClusterBuilder");
    declare_cluster_utility<double>(m);
    declare_interface_direction_computation<double>(m);
    declare_compute_largest_extent<double>(m);
    declare_compute_bounding_box<double>(m);
    declare_interface_splitting<double>(m);
    declare_regular_splitting<double>(m);
    declare_geometric_splitting<double>(m);

    declare_hmatrix_builder<double, double>(m, "HMatrixBuilder");
    declare_HMatrix<double, double>(m, "HMatrix");
    declare_virtual_generator<double>(m, "VirtualGenerator", "IGenerator");
    declare_custom_VirtualLowRankGenerator<double, double>(m, "VirtualLowRankGenerator");
    declare_custom_VirtualDenseBlocksGenerator<double>(m, "VirtualDenseBlocksGenerator");

    declare_local_operator<double, double>(m, "LocalOperator");
    declare_local_hmatrix<double, double>(m, "LocalHMatrix");

    declare_interface_partition<double>(m, "IPartition");
    declare_partition_from_cluster<double, double>(m, "PartitionFromCluster");

    declare_distributed_operator<double>(m, "DistributedOperator");
    declare_distributed_operator_utility<double, double>(m);

    declare_DDM<double>(m, "Solver");
    declare_solver_utility<double, double>(m);

    declare_matplotlib_cluster<double>(m);
    declare_matplotlib_hmatrix<double, double>(m);

    declare_HMatrix<std::complex<double>, double>(m, "ComplexHMatrix");
    declare_virtual_generator<std::complex<double>>(m, "ComplexVirtualGenerator", "IComplexGenerator");

    declare_distributed_operator<std::complex<double>>(m, "ComplexDistributedOperator");
    declare_distributed_operator_utility<std::complex<double>, double>(m, "Complex");

    declare_DDM<std::complex<double>>(m, "ComplexSolver");
    declare_solver_utility<std::complex<double>, double>(m, "Complex");
}
