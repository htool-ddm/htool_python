#ifndef HTOOL_PYBIND11_CLUSTERING_VIRTUAL_PARTITIONING_HPP
#define HTOOL_PYBIND11_CLUSTERING_VIRTUAL_PARTITIONING_HPP

#include <htool/clustering/interfaces/virtual_partitioning.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision>
void declare_virtual_partitioning(py::module &m, const std::string &prefix) {
    using Class = htool::VirtualPartitioning<CoefficientPrecision>;
    py::class_<Class, std::shared_ptr<Class>> py_class(m, (prefix + "VirtualPartitioning").c_str());
}
#endif
