#ifndef HTOOL_PYBIND11_CLUSTER_UTILITY_CPP
#define HTOOL_PYBIND11_CLUSTER_UTILITY_CPP

#include <htool/clustering/cluster_node.hpp>
#include <htool/clustering/cluster_output.hpp>
#include <pybind11/pybind11.h>

template <typename CoordinatePrecision>
void declare_cluster_utility(py::module &m) {
    m.def("read_cluster_from", &htool::read_cluster_tree<CoordinatePrecision>);
}

#endif
