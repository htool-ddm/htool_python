#ifndef HTOOL_PYBIND11_SOLVER_UTILITY_HPP
#define HTOOL_PYBIND11_SOLVER_UTILITY_HPP

#include <htool/solvers/utility.hpp>

void declare_solver_utility(py::module &m) {
    py::class_<LocalNumberingBuilder> py_class(m, "LocalNumberingBuilder");
    py_class.def(py::init<const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>());
    py_class.def_property_readonly(
        "local_to_global_numbering", [](const LocalNumberingBuilder &self) { return &self.local_to_global_numbering; }, py::return_value_policy::reference_internal);
    py_class.def_property_readonly(
        "intersections", [](const LocalNumberingBuilder &self) { return &self.intersections; }, py::return_value_policy::reference_internal);
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_solver_utility(py::module &m, std::string prefix = "") {

    using DefaultSolverBuilder                 = DefaultSolverBuilder<CoefficientPrecision, CoordinatePrecision>;
    using DefaultDDMSolverBuilderAddingOverlap = DefaultDDMSolverBuilderAddingOverlap<CoefficientPrecision, CoordinatePrecision>;
    using DefaultDDMSolverBuilder              = DefaultDDMSolverBuilder<CoefficientPrecision, CoordinatePrecision>;

    std::string default_solver_name                    = prefix + "DefaultSolverBuilder";
    std::string default_ddm_solver_adding_overlap_name = prefix + "DefaultDDMSolverBuilderAddingOverlap";
    std::string default_ddm_solver_name                = prefix + "DefaultDDMSolverBuilder";

    py::class_<DefaultSolverBuilder> default_solver_class(m, default_solver_name.c_str());
    default_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *>());
    default_solver_class.def_property_readonly(
        "solver", [](const DefaultSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);

    py::class_<DefaultDDMSolverBuilderAddingOverlap> default_ddm_solver_adding_overlap_class(m, default_ddm_solver_adding_overlap_name.c_str());
    default_ddm_solver_adding_overlap_class.def(py::init<DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> *, const VirtualGeneratorWithPermutation<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>());
    default_ddm_solver_adding_overlap_class.def_property_readonly(
        "solver", [](const DefaultDDMSolverBuilderAddingOverlap &self) { return &self.solver; }, py::return_value_policy::reference_internal);
    default_ddm_solver_adding_overlap_class.def_property_readonly(
        "local_to_global_numbering", [](const DefaultDDMSolverBuilderAddingOverlap &self) { return &self.local_to_global_numbering; }, py::return_value_policy::reference_internal);

    py::class_<DefaultDDMSolverBuilder> default_ddm_solver_class(m, default_ddm_solver_name.c_str());
    default_ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, const HMatrix<CoefficientPrecision, CoordinatePrecision> &, const std::vector<int> &, const std::vector<std::vector<int>> &>(), py::keep_alive<1, 2>(), py::keep_alive<1, 4>(), py::keep_alive<1, 5>());
    default_ddm_solver_class.def_property_readonly(
        "solver", [](const DefaultDDMSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);
}
#endif
