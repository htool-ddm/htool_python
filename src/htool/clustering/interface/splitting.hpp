#ifndef HTOOL_PYBIND11_CLUSTERING_INTERFACE_SPLITTING_HPP
#define HTOOL_PYBIND11_CLUSTERING_INTERFACE_SPLITTING_HPP

#include <htool/clustering/tree_builder/splitting.hpp>

template <typename CoordinatePrecision>
void declare_interface_splitting(py::module &m) {
    using Class = htool::VirtualSplittingStrategy<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>> py_class(m, "ISplitting");
}
#endif
