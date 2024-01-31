#ifndef HTOOL_PYTHON_VIRTUAL_COARSE_OPERATOR_BUILDER_HPP
#define HTOOL_PYTHON_VIRTUAL_COARSE_OPERATOR_BUILDER_HPP

#include <htool/solvers/interfaces/virtual_coarse_operator_builder.hpp>
#include <pybind11/pybind11.h>
namespace py = pybind11;

template <typename CoefficientPrecision>
void declare_virtual_coarse_operator_builder(py::module &m, const std::string &className, const std::string &base_class_name) {
    using BaseClass = VirtualCoarseOperatorBuilder<CoefficientPrecision>;
    py::class_<BaseClass>(m, base_class_name.c_str());
}

#endif
