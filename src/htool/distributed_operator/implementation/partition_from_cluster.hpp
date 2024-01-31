#ifndef HTOOL_DISTRIBUTED_OPERATOR_PARTITION_FROM_CLUSTER_CPP
#define HTOOL_DISTRIBUTED_OPERATOR_PARTITION_FROM_CLUSTER_CPP

#include <htool/distributed_operator/implementations/partition_from_cluster.hpp>
#include <htool/distributed_operator/interfaces/partition.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_partition_from_cluster(py::module &m, const std::string &class_name) {
    using Class = htool::PartitionFromCluster<CoefficientPrecision, CoordinatePrecision>;

    py::class_<Class, htool::IPartition<CoefficientPrecision>> py_class(m, class_name.c_str());
    py_class.def(py::init<const Cluster<CoordinatePrecision> &>(), py::keep_alive<1, 2>());
}

#endif
