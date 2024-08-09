#ifndef HTOOL_DDM_SOLVER_CPP
#define HTOOL_DDM_SOLVER_CPP

#include "htool/solvers/ddm.hpp"
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision>
void declare_DDM(py::module &m, const std::string &className) {

    using Class = DDM<CoefficientPrecision, HPDDMCustomLocalSolver>;
    py::class_<Class> py_class(m, className.c_str());
    py_class.def("facto_one_level", &Class::facto_one_level);
    py_class.def("build_coarse_space", py::overload_cast<VirtualCoarseSpaceBuilder<CoefficientPrecision> &, VirtualCoarseOperatorBuilder<CoefficientPrecision> &>(&Class::build_coarse_space));
    py_class.def(
        "solve", [](Class &self, py::array_t<CoefficientPrecision, py::array::f_style> x, const py::array_t<CoefficientPrecision, py::array::f_style | py::array::forcecast> b, std::string hpddm_args) {
            // HPDDM arguments
            HPDDM::Option &opt = *HPDDM::Option::get();
            opt.parse(hpddm_args);
            int mu;

            if (b.ndim() == 1 && x.ndim() == 1) {
                mu = 1;
            } else if ((b.ndim() == 2 && x.ndim() == 2) && b.shape()[1] == x.shape()[1]) {
                mu = b.shape()[1];
            } else {
                std::string rhs = "("; // LCOV_EXCL_START
                std::string sol = "(";
                for (int p = 0; p < b.ndim(); p++) {
                    rhs += htool::NbrToStr(b.shape()[p]);
                    if (p != b.ndim() - 1) {
                        rhs += ",";
                    }
                }
                rhs += ")";
                for (int p = 0; p < x.ndim(); p++) {
                    sol += htool::NbrToStr(x.shape()[p]);
                    if (p != x.ndim() - 1) {
                        sol += ",";
                    }
                }
                sol += ")";
                throw std::invalid_argument("Wrong dimension for right-hand side or solution\nright-hand side: " + rhs + "\n" + "solution: " + sol + "\n");
                // LCOV_EXCL_STOP
            }

            self.solve(b.data(), x.mutable_data(), mu);
        },
        py::arg("x").noconvert(true),
        py::arg("b"),
        py::arg("hpddm_args") = "");
    py_class.def("set_hpddm_args", [](Class &self, std::string hpddm_args) {
        HPDDM::Option &opt = *HPDDM::Option::get();
        opt.parse(hpddm_args);
    });
    py_class.def("get_information", py::overload_cast<>(&Class::get_information, py::const_));
}

#endif
