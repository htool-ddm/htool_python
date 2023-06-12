#ifndef HTOOL_CLUSTERING_CLUSTER_BUILDER_CPP
#define HTOOL_CLUSTERING_CLUSTER_BUILDER_CPP

#include "cluster_node.hpp"
#include <htool/clustering/tree_builder/direction_computation.hpp>
#include <htool/clustering/tree_builder/recursive_build.hpp>
#include <htool/clustering/tree_builder/splitting.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoordinatePrecision>
void declare_cluster_builder(py::module &m, const std::string &className) {

    using Class = ClusterTreeBuilder<CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());

    py_class.def(py::init<>());
    py_class.def("create_cluster_tree", [](Class &self, py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, int number_of_children, int size_of_partition) {
        return self.create_cluster_tree(coordinates.shape()[1], coordinates.shape()[0], coordinates.data(), number_of_children, size_of_partition);
    });
    py_class.def("create_cluster_tree", [](Class &self, py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, int number_of_children, int size_of_partition, py::array_t<int, py::array::f_style | py::array::forcecast> partition) {
        if (partition.ndim() != 2 && partition.shape()[0] != 2) {
            throw std::runtime_error("Wrong format for partition"); // LCOV_EXCL_LINE
        }
        return self.create_cluster_tree(coordinates.shape()[1], coordinates.shape()[0], coordinates.data(), number_of_children, size_of_partition, partition.data());
    });
    py_class.def("set_minclustersize", &Class::set_minclustersize);
    py_class.def("set_direction_computation_strategy", &Class::set_direction_computation_strategy);
    py_class.def("set_splitting_strategy", &Class::set_splitting_strategy);
}

#endif
