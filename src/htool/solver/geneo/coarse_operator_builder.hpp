#ifndef HTOOL_PYTHON_GENEO_COARSE_OPERATOR_BUILDER_HPP
#define HTOOL_PYTHON_GENEO_COARSE_OPERATOR_BUILDER_HPP

#include <htool/solvers/geneo/coarse_operator_builder.hpp>
#include <pybind11/pybind11.h>
namespace py = pybind11;

template <typename CoefficientPrecision>
void declare_geneo_coarse_operator_builder(py::module &m, const std::string &className) {

    using Class = GeneoCoarseOperatorBuilder<CoefficientPrecision>;
    py::class_<Class, VirtualCoarseOperatorBuilder<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init<const DistributedOperator<CoefficientPrecision> &>());
}

#endif
