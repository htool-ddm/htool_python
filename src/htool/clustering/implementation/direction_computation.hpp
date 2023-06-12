#ifndef HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_DIRECTION_COMPUTATION_HPP
#define HTOOL_PYBIND11_CLUSTERING_IMPLEMENTATION_DIRECTION_COMPUTATION_HPP

#include <htool/clustering/tree_builder/direction_computation.hpp>

template <typename CoordinatePrecision>
void declare_compute_largest_extent(py::module &m) {
    using Class = htool::ComputeLargestExtent<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, htool::VirtualDirectionComputationStrategy<CoordinatePrecision>> py_class(m, "ComputeLargestExtent");
    py_class.def(py::init<>());
}

template <typename CoordinatePrecision>
void declare_compute_bounding_box(py::module &m) {
    using Class = htool::ComputeBoundingBox<CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, htool::VirtualDirectionComputationStrategy<CoordinatePrecision>> py_class(m, "ComputeBoundingBox");
    py_class.def(py::init<>());
}
#endif
