#include "cluster.hpp"
#include "ddm_solver.hpp"
#include "hmatrix.hpp"
#include "matrix.hpp"
#include "wrapper_mpi.hpp"

PYBIND11_MODULE(Htool, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API.");
    }

    declare_IMatrix<double>(m, "IMatrix");
    declare_IMatrix<std::complex<double>>(m, "ComplexIMatrix");

    declare_Cluster<RegularClustering>(m, "Cluster");

    declare_HMatrix<double, sympartialACA, RegularClustering, RjasanowSteinbach>(m, "HMatrixVirtual", "HMatrix");
    declare_HMatrix<std::complex<double>, sympartialACA, RegularClustering, RjasanowSteinbach>(m, "ComplexHMatrixVirtual", "ComplexHMatrix");

    declare_DDM<double>(m, "DDM");
    declare_DDM<std::complex<double>>(m, "ComplexDDM");
}