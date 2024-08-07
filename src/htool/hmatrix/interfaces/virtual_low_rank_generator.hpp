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
    int m_rank;
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_U; // owned by Python
    mutable std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> m_mats_V; // owned by Python
    const VirtualGeneratorInUserNumbering<CoefficientPrecision> &m_generator_in_user_numbering;

  public:
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGenerator;

    VirtualLowRankGeneratorPython(const VirtualGeneratorInUserNumbering<CoefficientPrecision> &generator_in_user_numbering) : m_generator_in_user_numbering(generator_in_user_numbering) {}

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {
        py::array_t<int, py::array::f_style> rows(target_cluster.get_size(), target_cluster.get_permutation().data() + target_cluster.get_offset(), py::capsule(target_cluster.get_permutation().data()));
        py::array_t<int, py::array::f_style> cols(source_cluster.get_size(), source_cluster.get_permutation().data() + source_cluster.get_offset(), py::capsule(source_cluster.get_permutation().data()));

        build_low_rank_approximation(rows, cols, epsilon);
        rank = m_rank;
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
    void set_rank(int rank) { m_rank = rank; }

    void build_submatrix(const py::array_t<int> rows, const py::array_t<int> cols, py::array_t<CoefficientPrecision, py::array::f_style> &mat) const {
        m_generator_in_user_numbering.copy_submatrix(rows.size(), cols.size(), rows.data(), cols.data(), mat.mutable_data());
    }
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
    py_class.def(py::init<const VirtualGeneratorInUserNumbering<CoefficientPrecision> &>());
    py_class.def("build_low_rank_approximation", &Class::build_low_rank_approximation);
    py_class.def("set_U", &Class::set_U);
    py_class.def("set_V", &Class::set_V);
    py_class.def("set_rank", &Class::set_rank);
    py_class.def("build_submatrix", &Class::build_submatrix);
}

#endif
