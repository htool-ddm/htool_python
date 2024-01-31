#ifndef HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_PARTITION_HPP
#define HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_PARTITION_HPP

#include <htool/distributed_operator/interfaces/partition.hpp>
#include <pybind11/pybind11.h>

// class PyIPartition : public htool::IPartition {
//   public:
//     /* Inherit the constructors */
//     using htool::IPartition::IPartition;

//     /* Trampoline (need one for each virtual function) */
//     std::string go(int n_times) override {
//         PYBIND11_OVERRIDE_PURE(
//             std::string, /* Return type */
//             IPartition,  /* Parent class */
//             go,          /* Name of function in C++ (must match Python name) */
//             n_times      /* Argument(s) */
//         );
//     }
// };

template <typename CoefficientPrecision>
void declare_interface_partition(py::module &m, const std::string &class_name) {
    using Class = htool::IPartition<CoefficientPrecision>;
    py::class_<Class> py_class(m, class_name.c_str());
    py_class.def("get_size_of_partition", &Class::get_size_of_partition);
    py_class.def("get_offset_of_partition", &Class::get_offset_of_partition);
    py_class.def("get_global_size", &Class::get_global_size);
}

#endif
