#ifndef HTOOL_CLUSTER_CPP
#define HTOOL_CLUSTER_CPP
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <htool/clustering/cluster_node.hpp>
#include <mpi.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoordinatePrecision>
void declare_cluster_node(py::module &m, const std::string &className) {

    using Class = Cluster<CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());

    py_class.def("get_size", &Class::get_size);
    py_class.def("get_offset", &Class::get_offset);
    py_class.def("get_permutation", [](const Class &self) {
        auto &permutation = self.get_permutation();
        return py::array_t<int>(std::array<std::size_t, 1>{permutation.size()}, permutation.data(), py::capsule(permutation.data()));
        ;
    });
    py_class.def("get_cluster_on_partition", &Class::get_cluster_on_partition, py::return_value_policy::reference_internal);
}

#endif
