#ifndef HTOOL_PYBIND11_SOLVER_UTILITY_HPP
#define HTOOL_PYBIND11_SOLVER_UTILITY_HPP

#include <htool/solvers/utility.hpp>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_solver_utility(py::module &m, std::string prefix = "") {

    using DDMSolverBuilder = DDMSolverBuilder<CoefficientPrecision, CoordinatePrecision>;

    std::string ddm_solver_name = prefix + "DDMSolverBuilder";

    py::class_<DDMSolverBuilder> ddm_solver_class(m, ddm_solver_name.c_str());
    ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &>());

    ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &, const VirtualGeneratorInUserNumbering<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>());
    ddm_solver_class.def(py::init([](DistributedOperator<CoefficientPrecision> &distributed_operator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, const VirtualGeneratorInUserNumbering<CoefficientPrecision> &generator, const py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta) {
        return new DDMSolverBuilder(distributed_operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, generator, coordinates.shape()[0], coordinates.data(), epsilon, eta);
    }));

    ddm_solver_class.def_property_readonly(
        "solver", [](const DDMSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);
}
#endif
