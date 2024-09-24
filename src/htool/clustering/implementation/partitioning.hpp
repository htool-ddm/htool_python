#ifndef HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_PARTITIONING_HPP
#define HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_PARTITIONING_HPP

#include <htool/clustering/implementations/partitioning.hpp>

template <typename CoordinatePrecision, class ComputationDirectionPolicy, class SplittingPolicy>
void declare_partitioning(py::module &m, std::string name) {
    using Class = htool::Partitioning<CoordinatePrecision, ComputationDirectionPolicy, SplittingPolicy>;
    py::class_<Class, std::shared_ptr<Class>, VirtualPartitioning<CoordinatePrecision>> py_class(m, name.c_str());
    py_class.def(py::init<>());
}

#endif
