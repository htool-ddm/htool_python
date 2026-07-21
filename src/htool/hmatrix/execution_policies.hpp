#ifndef HTOOL_PYBIND11_EXECUTION_POLICIES_CPP
#define HTOOL_PYBIND11_EXECUTION_POLICIES_CPP

#include <htool/hmatrix/execution_policies.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace htool;

inline void declare_standard_policies(py::module &m) {
    py::class_<exec_compat::sequenced_policy>(
        m,
        "SequentialPolicy")
        .def(py::init<>());

    py::class_<exec_compat::parallel_policy>(
        m,
        "ParallelPolicy")
        .def(py::init<>());
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_omp_task_policy(py::module &m, std::string prefix = "") {
    py::class_<htool::omp_task_policy<CoefficientPrecision, CoordinatePrecision>>(
        m,
        (prefix + "OmpTaskPolicy").c_str())
        .def(py::init<>())
        .def_property(
            "max_number_of_nodes", [](htool::omp_task_policy<CoefficientPrecision, CoordinatePrecision> &self) { return self.hmatrix_task_dependencies.max_number_of_nodes; }, [](htool::omp_task_policy<CoefficientPrecision, CoordinatePrecision> &self, int value) { self.hmatrix_task_dependencies.max_number_of_nodes = value; })
        .def(
            "set_L0",
            [](
                htool::omp_task_policy<CoefficientPrecision, CoordinatePrecision> &self,

                HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
                self.hmatrix_task_dependencies.set_L0(hmatrix);
            },
            py::keep_alive<1, 2>())
        .def_property_readonly(
            "L0",
            [](htool::omp_task_policy<CoefficientPrecision, CoordinatePrecision> &self) {
                py::list result;
                for (HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix : self.hmatrix_task_dependencies.L0) {
                    result.append(py::cast(hmatrix, py::return_value_policy::reference));
                }

                return result;
            },
            py::return_value_policy::reference_internal);
}

#endif
