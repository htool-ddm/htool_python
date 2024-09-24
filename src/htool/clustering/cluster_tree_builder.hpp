#ifndef HTOOL_CLUSTERING_CLUSTER_BUILDER_CPP
#define HTOOL_CLUSTERING_CLUSTER_BUILDER_CPP

#include <htool/clustering/tree_builder/tree_builder.hpp>
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
    py_class.def("create_cluster_tree", [](Class &self, py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, int number_of_children, std::optional<int> size_of_partition, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> radii, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> weights) {
            int size_of_partition_final = size_of_partition.has_value() ? size_of_partition.value() : number_of_children;
            auto radii_ptr              = radii.has_value() ? radii.value().data() : nullptr;
            auto weights_ptr            = weights.has_value() ? weights.value().data() : nullptr;
            return self.create_cluster_tree(coordinates.shape()[1], coordinates.shape()[0], coordinates.data(), radii_ptr, weights_ptr, number_of_children, size_of_partition_final, nullptr, false); },
                 py::arg("coordinates"), // LCOV_EXCL_START
                 py::arg("number_of_children"),
                 py::kw_only(),
                 py::arg("size_of_partition") = py::none(),
                 py::arg("radii")             = py::none(),
                 py::arg("weights")           = py::none());
    // LCOV_EXCL_STOP

    py_class.def("create_cluster_tree_from_global_partition", [](Class &self, py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, int number_of_children, int size_of_partition, py::array_t<int, py::array::f_style | py::array::forcecast> partition, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> radii, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> weights) {
            if (partition.ndim() != 1 && partition.shape()[0] != coordinates.shape()[1]) {
                throw std::runtime_error("Wrong format for partition"); // LCOV_EXCL_LINE
            }

            auto radii_ptr   = radii.has_value() ? radii.value().data() : nullptr;
            auto weights_ptr = weights.has_value() ? weights.value().data() : nullptr;
            return self.create_cluster_tree(coordinates.shape()[1], coordinates.shape()[0], coordinates.data(), radii_ptr, weights_ptr, number_of_children, size_of_partition, partition.data(), false); },
                 py::arg("coordinates"), // LCOV_EXCL_START
                 py::arg("number_of_children"),
                 py::arg("size_of_partition"),
                 py::arg("partition"),
                 py::kw_only(),
                 py::arg("radii")   = py::none(),
                 py::arg("weights") = py::none());
    // LCOV_EXCL_STOP

    py_class.def("create_cluster_tree_from_local_partition", [](Class &self, py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, int number_of_children, int size_of_partition, py::array_t<int, py::array::f_style | py::array::forcecast> partition, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> radii, std::optional<py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> weights) {
            if (partition.ndim() != 2 && partition.shape()[0] != 2 && 2 * partition.shape()[1] != size_of_partition) {
                throw std::runtime_error("Wrong format for partition"); // LCOV_EXCL_LINE
            }

            auto radii_ptr   = radii.has_value() ? radii.value().data() : nullptr;
            auto weights_ptr = weights.has_value() ? weights.value().data() : nullptr;
            return self.create_cluster_tree(coordinates.shape()[1], coordinates.shape()[0], coordinates.data(), radii_ptr, weights_ptr, number_of_children, size_of_partition, partition.data(), true); },
                 py::arg("coordinates"), // LCOV_EXCL_START
                 py::arg("number_of_children"),
                 py::arg("size_of_partition"),
                 py::arg("partition"),
                 py::kw_only(),
                 py::arg("radii")   = py::none(),
                 py::arg("weights") = py::none());
    // LCOV_EXCL_STOP

    py_class.def("set_maximal_leaf_size", &Class::set_maximal_leaf_size);
    py_class.def("set_partitioning_strategy", &Class::set_partitioning_strategy);
}

#endif
