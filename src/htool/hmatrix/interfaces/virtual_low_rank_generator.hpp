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

    // py::array_t<CoefficientPrecision, py::array::f_style> m_mat_U, m_mat_V;

  public:
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {
        build_low_rank_approximation(A, target_cluster.get_size(), source_cluster.get_size(), target_cluster.get_offset(), source_cluster.get_offset(), epsilon);
        rank = m_rank;
        U.assign(m_mats_U.back().shape()[0], m_mats_U.back().shape()[1], m_mats_U.back().mutable_data(), false);
        V.assign(m_mats_V.back().shape()[0], m_mats_V.back().shape()[1], m_mats_V.back().mutable_data(), false);
    }

    bool is_htool_owning_data() const override { return false; }

    // lcov does not see it because of trampoline I assume
    virtual void build_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, int target_size, int source_size, int target_offset, int source_offset, underlying_type<CoefficientPrecision> epsilon) const = 0; // LCOV_EXCL_LINE

    void set_U(py::array_t<CoefficientPrecision, py::array::f_style> U0) {
        m_mats_U.push_back(U0); // no copy here
    }
    void set_V(py::array_t<CoefficientPrecision, py::array::f_style> V0) { m_mats_V.push_back(V0); }
    void set_rank(int rank) { m_rank = rank; }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class PyVirtualLowRankGenerator : public VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision> {
  public:
    using VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGeneratorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void build_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, int target_size, int source_size, int target_offset, int source_offset, underlying_type<CoefficientPrecision> epsilon) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                /* Return type */
            VirtualLowRankGeneratorPython<CoefficientPrecision>, /* Parent class */
            build_low_rank_approximation,                        /* Name of function in C++ (must match Python name) */
            A,
            target_size,
            source_size,
            target_offset,
            source_offset,
            epsilon);
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_custom_VirtualLowRankGenerator(py::module &m, const std::string &className) {
    // using BaseClass = VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>;
    // py::class_<BaseClass, std::shared_ptr<BaseClass>>(m, base_class_name.c_str());

    using Class = VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> py_class(m, className.c_str());
    py_class.def(py::init<>());
    py_class.def("build_low_rank_approximation", &Class::build_low_rank_approximation);
    py_class.def("set_U", &Class::set_U);
    py_class.def("set_V", &Class::set_V);
    py_class.def("set_rank", &Class::set_rank);
}

// template <typename T>
// void declare_VirtualLowRankGenerator(py::module &m, const std::string &className) {
//     using Class = VirtualLowRankGenerator<T>;
//     py::class_<Class> py_class(m, className.c_str());
// }

// template <template <class> class LowRankGeneratorType, typename T>
// void declare_predefined_LowRankGenerator(py::module &m, const std::string &className) {

//     py::class_<LowRankGeneratorType<T>, std::shared_ptr<LowRankGeneratorType<T>>, VirtualLowRankGenerator<T>> py_class(m, className.c_str());
//     py_class.def(py::init<>());
//     py_class.def("copy_low_rank_approximation", &LowRankGeneratorType<T>::copy_low_rank_approximation);
// }

#endif
