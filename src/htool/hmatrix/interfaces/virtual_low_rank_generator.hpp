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

template <typename CoefficientPrecision>
class VirtualLowRankGeneratorPython : public VirtualLowRankGenerator<CoefficientPrecision> {
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_U; // owned by Python
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_V; // owned by Python
    bool m_allow_copy = true;

  public:
    using VirtualLowRankGenerator<CoefficientPrecision>::VirtualLowRankGenerator;

    VirtualLowRankGeneratorPython(bool allow_copy = true) : VirtualLowRankGenerator<CoefficientPrecision>(), m_allow_copy(allow_copy) {}

    bool copy_low_rank_approximation(int M, int N, const int *const rows, const int *const cols, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        py::gil_scoped_acquire acquire;
        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        py::array_t<int> py_rows(std::array<long int, 1>{M}, rows, py::capsule(rows));
        py::array_t<int> py_cols(std::array<long int, 1>{N}, cols, py::capsule(cols));
        bool success = build_low_rank_approximation(py_rows, py_cols, lrmat.get_epsilon());
        if (success) {
            if (m_allow_copy) {
                U.resize(m_mats_U.back().shape()[0], m_mats_U.back().shape()[1]);
                std::copy_n(m_mats_U.back().mutable_data(), U.nb_rows() * U.nb_cols(), U.data());
                V.resize(m_mats_V.back().shape()[0], m_mats_V.back().shape()[1]);
                std::copy_n(m_mats_V.back().mutable_data(), V.nb_rows() * V.nb_cols(), V.data());
                m_mats_V.pop_back();
            } else {
                U.assign(m_mats_U.back().shape()[0], m_mats_U.back().shape()[1], m_mats_U.back().mutable_data(), false);
                V.assign(m_mats_V.back().shape()[0], m_mats_V.back().shape()[1], m_mats_V.back().mutable_data(), false);
            }
        }
        return success;
    }

    bool copy_low_rank_approximation(int M, int N, const int *const rows, const int *const cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override { // LCOV_EXCL_LINE
        Logger::get_instance().log(LogLevel::ERROR, "copy_low_rank_approximation with required rank is not implemented in the python interface.");                         // LCOV_EXCL_LINE
        return false;                                                                                                                                                      // LCOV_EXCL_LINE
    }

    // lcov does not see it because of trampoline I assume
    virtual bool build_low_rank_approximation(const py::array_t<int, py::array::f_style> &rows, const py::array_t<int, py::array::f_style> &cols, underlying_type<CoefficientPrecision> epsilon) const = 0; // LCOV_EXCL_LINE

    void set_U(py::array_t<CoefficientPrecision, py::array::f_style> U0) {
        m_mats_U.push_back(U0); // no copy here
    }
    void set_V(py::array_t<CoefficientPrecision, py::array::f_style> V0) { m_mats_V.push_back(V0); }

    void clear_data() {
        m_mats_U.clear();
        m_mats_V.clear();
    }
};

template <typename CoefficientPrecision>
class PyVirtualLowRankGenerator : public VirtualLowRankGeneratorPython<CoefficientPrecision> {
  public:
    using VirtualLowRankGeneratorPython<CoefficientPrecision>::VirtualLowRankGeneratorPython;

    /* Trampoline (need one for each virtual function) */
    virtual bool build_low_rank_approximation(const py::array_t<int, py::array::f_style> &rows, const py::array_t<int, py::array::f_style> &cols, underlying_type<CoefficientPrecision> epsilon) const override {
        PYBIND11_OVERRIDE_PURE(
            bool,                                                /* Return type */
            VirtualLowRankGeneratorPython<CoefficientPrecision>, /* Parent class */
            build_low_rank_approximation,                        /* Name of function in C++ (must match Python name) */
            rows,
            cols,
            epsilon);
    }
};

template <typename CoefficientPrecision>
void declare_custom_VirtualLowRankGenerator(py::module &m, const std::string &className) {
    using Class = VirtualLowRankGeneratorPython<CoefficientPrecision>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualLowRankGenerator<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init<bool>(), "allow_copy"_a = true);
    py_class.def("build_low_rank_approximation", &Class::build_low_rank_approximation);
    py_class.def("set_U", &Class::set_U);
    py_class.def("set_V", &Class::set_V);
    py_class.def("clear_data", &Class::clear_data);
}

#endif
