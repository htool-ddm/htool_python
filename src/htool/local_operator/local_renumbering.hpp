#ifndef HTOOL_VIRTUAL_LOCAL_RENUMBERING_CPP
#define HTOOL_VIRTUAL_LOCAL_RENUMBERING_CPP

#include <htool/distributed_operator/local_renumbering.hpp>
#include <pybind11/pybind11.h>

template <typename CoordinatePrecision>
void declare_local_renumbering(py::module &m, const std::string &className) {

    using Class = LocalRenumbering;
    py::class_<Class> py_class(m, className.c_str());
    py_class.def(py::init([](int offset, int size, py::array_t<int> permutation) {
        return std::unique_ptr<Class>(new Class(offset, size, permutation.size(), permutation.data()));
    }));
    py_class.def(py::init<const Cluster<CoordinatePrecision> &>());
    py_class.def_property_readonly("offset", &Class::get_offset);
    py_class.def_property_readonly("size", &Class::get_size);
    py_class.def_property_readonly("global_size", &Class::get_global_size);
    py_class.def_property_readonly("is_stable", &Class::is_stable);
    py_class.def_property_readonly("permutation", [](const Class &self) { return py::array_t<int>(std::array<long int, 1>{self.get_global_size()}, self.get_permutation(), py::capsule(self.get_permutation())); });
}

#endif
