#ifndef HTOOL_LRMAT_GENERATOR_CPP
#define HTOOL_LRMAT_GENERATOR_CPP

#include <htool/htool.hpp>
#include <htool/lrmat/virtual_lrmat_generator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename T>
class VirtualLowRankGeneratorCpp : public VirtualLowRankGenerator<T> {

    py::array_t<T, py::array::f_style> mat_U, mat_V;
    int rank;

  public:
    using VirtualLowRankGenerator<T>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank0, T **U, T **V, const VirtualGenerator<T> &A, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) const override {

        build_low_rank_approximation(epsilon, rank0, A, std::vector<int>(rows, rows + M), std::vector<int>(cols, cols + N));
        *U    = new T[M * rank];
        *V    = new T[N * rank];
        rank0 = rank;
        std::copy_n(mat_U.data(), mat_U.size(), *U);
        std::copy_n(mat_V.data(), mat_V.size(), *V);
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_low_rank_approximation(double epsilon, int rank, const VirtualGenerator<T> &A, const std::vector<int> &J, const std::vector<int> &K) const = 0; // LCOV_EXCL_LINE

    void set_U(py::array_t<T, py::array::f_style> U0) { mat_U = U0; }
    void set_V(py::array_t<T, py::array::f_style> V0) { mat_V = V0; }
    void set_rank(int rank0) { rank = rank0; }
};

template <typename T>
class PyVirtualLowRankGenerator : public VirtualLowRankGeneratorCpp<T> {
  public:
    using VirtualLowRankGeneratorCpp<T>::VirtualLowRankGeneratorCpp;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_low_rank_approximation(double epsilon, int rank, const VirtualGenerator<T> &A, const std::vector<int> &J, const std::vector<int> &K) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                          /* Return type */
            VirtualLowRankGeneratorCpp<T>, /* Parent class */
            build_low_rank_approximation,  /* Name of function in C++ (must match Python name) */
            epsilon,
            rank,
            A,
            J,
            K /* Argument(s) */
        );
    }
};

template <typename T>
void declare_custom_VirtualLowRankGenerator(py::module &m, const std::string &className) {
    using Class = VirtualLowRankGeneratorCpp<T>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualLowRankGenerator<T>> py_class(m, className.c_str());
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