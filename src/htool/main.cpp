#include "cluster.hpp"
#include "ddm_solver.hpp"
#include "hmatrix.hpp"
#include "matrix.hpp"

PYBIND11_MODULE(Htool, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
        throw std::runtime_error("Could not load mpi4py API.");
    }

    m.def("SetEpsilon", &SetEpsilon);
    m.def("SetEta", &SetEta);
    m.def("SetMinClusterSize", &SetMinClusterSize);
    m.def("SetMaxBlockSize", &SetMaxBlockSize);
    m.def("SetMinTargetDepth", &SetMinTargetDepth);
    m.def("SetMinSourceDepth", &SetMinSourceDepth);

    declare_IMatrix<double>(m, "IMatrix");
    declare_IMatrix<std::complex<double>>(m, "ComplexIMatrix");

    declare_Cluster<GeometricClusteringDDM>(m, "Cluster");

    declare_SubMatrix<double>(m, "SubMatrix");
    declare_SubMatrix<std::complex<double>>(m, "ComplexSubMatrix");

    declare_HMatrix<double, sympartialACA, GeometricClusteringDDM, RjasanowSteinbach>(m, "HMatrix");
    declare_HMatrix<std::complex<double>, sympartialACA, GeometricClusteringDDM, RjasanowSteinbach>(m, "ComplexHMatrix");

    declare_DDM<double, sympartialACA, GeometricClusteringDDM, RjasanowSteinbach>(m, "DDM");
    declare_DDM<std::complex<double>, sympartialACA, GeometricClusteringDDM, RjasanowSteinbach>(m, "ComplexDDM");
}