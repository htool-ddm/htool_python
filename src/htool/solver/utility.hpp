#ifndef HTOOL_PYBIND11_SOLVER_UTILITY_HPP
#define HTOOL_PYBIND11_SOLVER_UTILITY_HPP

#include <htool/solvers/utility.hpp>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_solver_utility(py::module &m, std::string prefix = "") {

    using DefaultSolverBuilder          = DefaultSolverBuilder<CoefficientPrecision, CoordinatePrecision>;
    using DefaultDDMSolverBuilder       = DefaultDDMSolverBuilder<CoefficientPrecision, CoordinatePrecision>;
    std::string default_solver_name     = prefix + "DefaultSolverBuilder";
    std::string default_ddm_solver_name = prefix + "DefaultDDMSolverBuilder";
    py::class_<DefaultSolverBuilder> default_solver_class(m, default_solver_name.c_str());
    default_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *>());
    default_solver_class.def_property_readonly(
        "solver", [](const DefaultSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);

    py::class_<DefaultDDMSolverBuilder> default_ddm_solver_class(m, default_ddm_solver_name.c_str());
    default_ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *, const VirtualGeneratorWithPermutation<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>());
    default_ddm_solver_class.def_property_readonly(
        "solver", [](const DefaultDDMSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);
}
#endif
