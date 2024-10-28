#ifndef HTOOL_LRMAT_CPP
#define HTOOL_LRMAT_CPP

#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace htool;

template <typename CoefficientPrecision>
void declare_LowRankMatrix(py::module &m, const std::string &className) {
    using Class = LowRankMatrix<CoefficientPrecision>;
    py::class_<Class> py_class(m, className.c_str());

    py_class.def("nb_rows", &Class::nb_rows);
    py_class.def("nb_cols", &Class::nb_cols);
    py_class.def("rank", &Class::rank_of, "test");
}
#endif
