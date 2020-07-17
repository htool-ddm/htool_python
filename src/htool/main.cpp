#include "matrix.hpp"
#include "hmatrix.hpp"


PYBIND11_MODULE(Htool, m) {
    // import the mpi4py API
    if (import_mpi4py() < 0) {
      throw std::runtime_error("Could not load mpi4py API.");
    }

    m.def("SetEpsilon", &SetEpsilon);
    m.def("SetEta", &SetEta);
    m.def("SetMinClusterSize", &SetMinClusterSize);


    declare_IMatrix<double>(m,"IMatrix");
    declare_IMatrix<std::complex<double>>(m,"ComplexIMatrix");
    declare_SubMatrix<double>(m,"SubMatrix");
    declare_SubMatrix<std::complex<double>>(m,"ComplexSubMatrix");
    declare_HMatrix<double,sympartialACA,GeometricClustering>(m, "HMatrix");
    declare_HMatrix<std::complex<double>,sympartialACA,GeometricClustering>(m, "ComplexHMatrix");
}