#ifndef HTOOL_VIRTUAL_LOCAL_OPERATOR_CPP
#define HTOOL_VIRTUAL_LOCAL_OPERATOR_CPP

#include <htool/local_operators/virtual_local_operator.hpp>
#include <pybind11/pybind11.h>

// template <typename CoefficientPrecision>
// class PyVirtualLocalOperator : public htool::VirtualLocalOperator<CoefficientPrecision> {
//   public:
// };

template <typename CoefficientPrecision>
void declare_interface_local_operator(py::module &m, const std::string &class_name) {
    using Class = htool::VirtualLocalOperator<CoefficientPrecision>;
    py::class_<Class>(m, class_name.c_str());
}

#endif
