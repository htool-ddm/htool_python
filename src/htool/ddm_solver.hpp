#ifndef HTOOL_DDM_SOLVER_CPP
#define HTOOL_DDM_SOLVER_CPP

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "htool/solvers/ddm.hpp"
#include "matrix.hpp"
#include "wrapper_mpi.hpp"
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename T>
void declare_DDM(py::module &m, const std::string &className) {

    using Class = DDM<T>;
    py::class_<Class> py_class(m, className.c_str());
    py_class.def(py::init<const VirtualHMatrix<T> *const>());
    py_class.def(py::init<
                 const VirtualGeneratorCpp<T> &,
                 const VirtualHMatrix<T> *const,
                 const std::vector<int> &,
                 const std::vector<int> &,
                 const std::vector<int> &,
                 const std::vector<std::vector<int>> &>());
    py_class.def("facto_one_level", &Class::facto_one_level);
    py_class.def(
        "solve", [](Class &self, py::array_t<T, py::array::f_style> x, const py::array_t<T, py::array::f_style | py::array::forcecast> b, std::string hpddm_args) {
            int rank;
            MPI_Comm_rank(self.get_comm(), &rank);

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
            if (b.shape()[0] != self.get_nb_cols()) {
                throw std::invalid_argument("Wrong size for right-hand side"); // LCOV_EXCL_LINE
            }
            if (x.shape()[0] != self.get_nb_rows()) {
                throw std::invalid_argument("Wrong size for solution"); // LCOV_EXCL_LINE
            }

            self.solve(b.data(), x.mutable_data(), mu);
        },
        py::arg("x").noconvert(true),
        py::arg("b"),
        py::arg("hpddm_args") = "");
    py_class.def("set_hpddm_args", [](Class &self, std::string hpddm_args) {
        int rank;
        MPI_Comm_rank(self.get_comm(), &rank);
        HPDDM::Option &opt = *HPDDM::Option::get();
        opt.parse(hpddm_args);
    });
    py_class.def("print_infos", &Class::print_infos);
    py_class.def("get_infos", &Class::get_infos);
}

#endif