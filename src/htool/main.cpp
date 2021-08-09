#include "cluster.hpp"
#include "ddm_solver.hpp"
#include "hmatrix.hpp"
#include "matrix.hpp"
#include "wrapper_mpi.hpp"

PYBIND11_MODULE(Htool, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API."); // LCOV_EXCL_LINE
    }

    declare_VirtualGenerator<double>(m, "IMatrix");
    declare_VirtualGenerator<std::complex<double>>(m, "ComplexIMatrix");

    py::class_<VirtualCluster, std::shared_ptr<VirtualCluster>>(m, "VirtualCluster");
    declare_Cluster<Cluster<PCARegularClustering>>(m, "PCARegularClustering");
    declare_Cluster<Cluster<PCAGeometricClustering>>(m, "PCAGeometricClustering");
    declare_Cluster<Cluster<BoundingBox1RegularClustering>>(m, "BoundingBox1RegularClustering");
    declare_Cluster<Cluster<BoundingBox1GeometricClustering>>(m, "BoundingBox1GeometricClustering");

    declare_HMatrix<double, sympartialACA, RjasanowSteinbach>(m, "HMatrixVirtual", "HMatrix");
    declare_HMatrix<std::complex<double>, sympartialACA, RjasanowSteinbach>(m, "ComplexHMatrixVirtual", "ComplexHMatrix");

    declare_DDM<double>(m, "DDM");
    declare_DDM<std::complex<double>>(m, "ComplexDDM");
}