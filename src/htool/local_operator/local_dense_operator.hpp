#ifndef HTOOL_LOCAL_DENSE_OPERATOR_CPP
#define HTOOL_LOCAL_DENSE_OPERATOR_CPP

#include "local_operator.hpp"
#include <htool/local_operators/local_dense_matrix.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_local_dense_matrix(py::module &m, const std::string &class_name) {
    using Class = htool::LocalDenseMatrix<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class, htool::LocalOperator<CoefficientPrecision, CoordinatePrecision>> py_data(m, class_name.c_str());
    py_data.def(py::init<const VirtualGeneratorPython<CoefficientPrecision> &, const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, char, char, bool, bool, int>());
}

#endif
