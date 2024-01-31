#ifndef HTOOL_PYBIND11_CLUSTERING_INTERFACE_DIRECTION_COMPUTATION_HPP
#define HTOOL_PYBIND11_CLUSTERING_INTERFACE_DIRECTION_COMPUTATION_HPP

#include <htool/clustering/tree_builder/direction_computation.hpp>

template <typename CoordinatePrecision>
void declare_interface_direction_computation(py::module &m) {
    using Class = htool::VirtualDirectionComputationStrategy<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>> py_class(m, "IDirectionComputation");
}
#endif
