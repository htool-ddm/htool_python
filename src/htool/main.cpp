#include "cluster.hpp"
#include "ddm_solver.hpp"
#include "dense_blocks_generator.hpp"
#include "hmatrix.hpp"
#include "lrmat_generator.hpp"
#include "matrix.hpp"
#include "off_diagonal_approximation.hpp"
#include "wrapper_mpi.hpp"

PYBIND11_MODULE(Htool, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API."); // LCOV_EXCL_LINE
    }

    declare_VirtualGenerator<double>(m, "VirtualGenerator");
    declare_VirtualGenerator<std::complex<double>>(m, "ComplexVirtualGenerator");

    py::class_<VirtualCluster, std::shared_ptr<VirtualCluster>>(m, "VirtualCluster");
    declare_Cluster<Cluster<PCARegularClustering>>(m, "PCARegularClustering");
    declare_Cluster<Cluster<PCAGeometricClustering>>(m, "PCAGeometricClustering");
    declare_Cluster<Cluster<BoundingBox1RegularClustering>>(m, "BoundingBox1RegularClustering");
    declare_Cluster<Cluster<BoundingBox1GeometricClustering>>(m, "BoundingBox1GeometricClustering");

    declare_custom_VirtualLowRankGenerator<double>(m, "CustomLowRankGenerator");
    declare_custom_VirtualDenseBlocksGenerator<double>(m, "CustomDenseBlocksGenerator");

    declare_HMatrix<double>(m, "HMatrixVirtual", "HMatrix");
    declare_HMatrix<std::complex<double>>(m, "ComplexHMatrixVirtual", "ComplexHMatrix");

    declare_VirtualOffDiagonalApproximation<double>(m, "VirtualOffDiagonalApproximation", "CustomOffDiagonalApproximation", "HMatrixOffDiagonalApproximation");
    declare_VirtualOffDiagonalApproximation<std::complex<double>>(m, "ComplexVirtualOffDiagonalApproximation", "ComplexCustomOffDiagonalApproximation", "ComplexHMatrixOffDiagonalApproximation");

    declare_DDM<double>(m, "DDM");
    declare_DDM<std::complex<double>>(m, "ComplexDDM");
}
