#ifndef HTOOL_LRMAT_GENERATOR_CPP
#define HTOOL_LRMAT_GENERATOR_CPP

#include <htool/hmatrix/interfaces/virtual_lrmat_generator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class VirtualLowRankGeneratorPython : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision> {
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_U; // owned by Python
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_V; // owned by Python

  public:
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGenerator;

    VirtualLowRankGeneratorPython() {}

    void copy_low_rank_approximation(const VirtualInternalGenerator<CoefficientPrecision> &A, int M, int N, const int *const rows, const int *const cols, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {
        py::array_t<int> py_rows(std::array<long int, 1>{M}, rows, py::capsule(rows));
        py::array_t<int> py_cols(std::array<long int, 1>{N}, cols, py::capsule(cols));

        build_low_rank_approximation(py_rows, py_cols, epsilon);
        U.assign(m_mats_U.back().shape()[0], m_mats_U.back().shape()[1], m_mats_U.back().mutable_data(), false);
        V.assign(m_mats_V.back().shape()[0], m_mats_V.back().shape()[1], m_mats_V.back().mutable_data(), false);
    }

    bool is_htool_owning_data() const override { return false; }

    // lcov does not see it because of trampoline I assume
    virtual void build_low_rank_approximation(const py::array_t<int, py::array::f_style> &rows, const py::array_t<int, py::array::f_style> &cols, underlying_type<CoefficientPrecision> epsilon) const = 0; // LCOV_EXCL_LINE

    void set_U(py::array_t<CoefficientPrecision, py::array::f_style> U0) {
        m_mats_U.push_back(U0); // no copy here
    }
    void set_V(py::array_t<CoefficientPrecision, py::array::f_style> V0) { m_mats_V.push_back(V0); }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class PyVirtualLowRankGenerator : public VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision> {
  public:
    using VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGeneratorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void build_low_rank_approximation(const py::array_t<int, py::array::f_style> &rows, const py::array_t<int, py::array::f_style> &cols, underlying_type<CoefficientPrecision> epsilon) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                /* Return type */
            VirtualLowRankGeneratorPython<CoefficientPrecision>, /* Parent class */
            build_low_rank_approximation,                        /* Name of function in C++ (must match Python name) */
            rows,
            cols,
            epsilon);
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_custom_VirtualLowRankGenerator(py::module &m, const std::string &className) {
    using Class = VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> py_class(m, className.c_str());
    py_class.def(py::init<>());
    py_class.def("build_low_rank_approximation", &Class::build_low_rank_approximation);
    py_class.def("set_U", &Class::set_U);
    py_class.def("set_V", &Class::set_V);
}

#endif
