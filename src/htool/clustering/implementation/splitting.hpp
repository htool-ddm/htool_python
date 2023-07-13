#ifndef HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_SPLITTING_HPP
#define HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_SPLITTING_HPP

#include <htool/clustering/tree_builder/splitting.hpp>

template <typename CoordinatePrecision>
void declare_regular_splitting(py::module &m) {
    using Class = htool::RegularSplitting<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, htool::VirtualSplittingStrategy<CoordinatePrecision>> py_class(m, "RegularSplitting");
    py_class.def(py::init<>());
}

template <typename CoordinatePrecision>
void declare_geometric_splitting(py::module &m) {
    using Class = htool::GeometricSplitting<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, htool::VirtualSplittingStrategy<CoordinatePrecision>> py_class(m, "GeometricSplitting");
    py_class.def(py::init<>());
}
#endif
