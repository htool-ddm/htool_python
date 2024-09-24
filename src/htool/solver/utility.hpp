#ifndef HTOOL_PYBIND11_SOLVER_UTILITY_HPP
#define HTOOL_PYBIND11_SOLVER_UTILITY_HPP

#include <htool/solvers/utility.hpp>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_solver_utility(py::module &m, std::string prefix = "") {

    using DDMSolverBuilder = DDMSolverBuilder<CoefficientPrecision, CoordinatePrecision>;

    std::string ddm_solver_name = prefix + "DDMSolverBuilder";

    py::class_<DDMSolverBuilder> ddm_solver_class(m, ddm_solver_name.c_str());
    ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &>());

    ddm_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &, const VirtualGenerator<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>());

    ddm_solver_class.def(py::init([](DistributedOperator<CoefficientPrecision> &distributed_operator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, const VirtualGenerator<CoefficientPrecision> &generator, const py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, const ClusterTreeBuilder<CoordinatePrecision> &cluster_tree_builder, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &local_hmatrix_builder, std::optional<const py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> radii, std::optional<const py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast>> weights) {
                             auto radii_ptr   = radii.has_value() ? radii.value().data() : nullptr;
                             auto weights_ptr = weights.has_value() ? weights.value().data() : nullptr;
                             return new DDMSolverBuilder(distributed_operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, generator, coordinates.shape()[0], coordinates.data(), radii_ptr, weights_ptr, cluster_tree_builder, local_hmatrix_builder);
                         }),
                         py::keep_alive<1, 2>(), // LCOV_EXCL_START
                         py::arg("distributed_operator"),
                         py::arg("ovr_subdomain_to_global"),
                         py::arg("cluster_to_ovr_subdomain"),
                         py::arg("neighbors"),
                         py::arg("intersections"),
                         py::arg("generator"),
                         py::arg("coordinates"),
                         py::arg("cluster_tree_builder"),
                         py::arg("local_hmatrix_builder"),
                         py::kw_only(),
                         py::arg("radii")   = py::none(),
                         py::arg("weights") = py::none());
    // LCOV_EXCL_STOP

    ddm_solver_class.def_property_readonly(
        "solver", [](const DDMSolverBuilder &self) { return &self.solver; }, py::return_value_policy::reference_internal);
    ddm_solver_class.def_property_readonly("local_to_global_numbering", [](const DDMSolverBuilder &self) { return &self.local_to_global_numbering; }, py::return_value_policy::reference_internal);
    ddm_solver_class.def(
        "get_local_hmatrix", [](const DDMSolverBuilder &self) { return self.local_hmatrix.get(); }, py::return_value_policy::reference_internal);

    using DDMSolverWithDenseLocalSolver = DDMSolverWithDenseLocalSolver<CoefficientPrecision, CoordinatePrecision>;

    std::string ddm_dense_solver_name = prefix + "DDMSolverWithDenseLocalSolver";

    py::class_<DDMSolverWithDenseLocalSolver> ddm_dense_solver_class(m, ddm_dense_solver_name.c_str());
    ddm_dense_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &>());

    ddm_dense_solver_class.def(py::init<DistributedOperator<CoefficientPrecision> &, HMatrix<CoefficientPrecision, CoordinatePrecision> &, const VirtualGenerator<CoefficientPrecision> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const std::vector<std::vector<int>> &>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>());
    ddm_dense_solver_class.def(py::init([](DistributedOperator<CoefficientPrecision> &distributed_operator, const std::vector<int> &ovr_subdomain_to_global, const std::vector<int> &cluster_to_ovr_subdomain, const std::vector<int> &neighbors, const std::vector<std::vector<int>> &intersections, const VirtualGenerator<CoefficientPrecision> &generator, const py::array_t<CoordinatePrecision, py::array::f_style | py::array::forcecast> coordinates, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &local_hmatrix_builder) {
                                   return new DDMSolverWithDenseLocalSolver(distributed_operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, generator, coordinates.shape()[0], coordinates.data(), local_hmatrix_builder);
                               }),
                               py::keep_alive<1, 2>());

    ddm_dense_solver_class.def_property_readonly(
        "solver", [](const DDMSolverWithDenseLocalSolver &self) { return &self.solver; }, py::return_value_policy::reference_internal);
    ddm_dense_solver_class.def_property_readonly("local_to_global_numbering", [](const DDMSolverWithDenseLocalSolver &self) { return &self.local_to_global_numbering; }, py::return_value_policy::reference_internal);
    ddm_dense_solver_class.def(
        "get_local_hmatrix", [](const DDMSolverWithDenseLocalSolver &self) { return self.local_hmatrix.get(); }, py::return_value_policy::reference_internal);
}
#endif
