#ifndef HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_UTILITY_HPP
#define HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_UTILITY_HPP

#include "../hmatrix/interfaces/virtual_generator.hpp"
#include "distributed_operator.hpp"
#include <htool/distributed_operator/utility.hpp>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_distributed_operator_utility(py::module &m, std::string prefix = "") {

    using DefaultApproximation                   = DefaultApproximationBuilder<CoefficientPrecision, CoordinatePrecision>;
    using LocalDefaultApproximation              = DefaultLocalApproximationBuilder<CoefficientPrecision, CoordinatePrecision>;
    std::string default_approximation_name       = prefix + "DefaultApproximationBuilder";
    std::string default_local_approximation_name = prefix + "DefaultLocalApproximationBuilder";
    py::class_<DefaultApproximation> default_approximation_class(m, default_approximation_name.c_str());
    default_approximation_class.def(py::init<const VirtualGenerator<CoefficientPrecision> &, const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, htool::underlying_type<CoefficientPrecision>, htool::underlying_type<CoefficientPrecision>, char, char, MPI_Comm_wrapper>());
    default_approximation_class.def_property_readonly(
        "distributed_operator", [](const DefaultApproximation &self) { return &self.distributed_operator; }, py::return_value_policy::reference_internal);
    default_approximation_class.def_property_readonly(
        "hmatrix", [](const DefaultApproximation &self) { return &self.hmatrix; }, py::return_value_policy::reference_internal);
    default_approximation_class.def_property_readonly(
        "block_diagonal_hmatrix", [](const DefaultApproximation &self) { return &*self.block_diagonal_hmatrix; }, py::return_value_policy::reference_internal);

    py::class_<LocalDefaultApproximation> default_local_approximation_class(m, default_local_approximation_name.c_str());
    default_local_approximation_class.def(py::init<const VirtualGenerator<CoefficientPrecision> &, const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, htool::underlying_type<CoefficientPrecision>, htool::underlying_type<CoefficientPrecision>, char, char, MPI_Comm_wrapper>());
    default_local_approximation_class.def_property_readonly(
        "distributed_operator", [](const LocalDefaultApproximation &self) { return &self.distributed_operator; }, py::return_value_policy::reference_internal);
    default_local_approximation_class.def_property_readonly(
        "hmatrix", [](const LocalDefaultApproximation &self) { return &self.hmatrix; }, py::return_value_policy::reference_internal);
    default_local_approximation_class.def_property_readonly(
        "block_diagonal_hmatrix", [](const LocalDefaultApproximation &self) { return &*self.block_diagonal_hmatrix; }, py::return_value_policy::reference_internal);
}
#endif
