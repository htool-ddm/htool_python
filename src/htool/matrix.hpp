#ifndef HTOOL_MATRIX_CPP
#define HTOOL_MATRIX_CPP

#include <htool/htool.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace htool;

template <typename T>
class VirtualGeneratorCpp : public VirtualGenerator<T> {
  public:
    using VirtualGenerator<T>::VirtualGenerator;

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {

        py::array_t<T, py::array::f_style> mat(std::array<long int, 2>{M, N}, ptr, py::capsule(ptr));
        build_submatrix(std::vector<int>(rows, rows + M), std::vector<int>(cols, cols + N), mat);
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_submatrix(const std::vector<int> &J, const std::vector<int> &K, py::array_t<T, py::array::f_style> &mat) const = 0; // LCOV_EXCL_LINE
};

template <typename T>
class PyVirtualGenerator : public VirtualGeneratorCpp<T> {
  public:
    using VirtualGeneratorCpp<T>::VirtualGeneratorCpp;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_submatrix(const std::vector<int> &J, const std::vector<int> &K, py::array_t<T, py::array::f_style> &mat) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                   /* Return type */
            VirtualGeneratorCpp<T>, /* Parent class */
            build_submatrix,        /* Name of function in C++ (must match Python name) */
            J,
            K,
            mat /* Argument(s) */
        );
    }
};

template <typename T>
void declare_VirtualGenerator(py::module &m, const std::string &className) {
    using Class = VirtualGeneratorCpp<T>;
    py::class_<Class, PyVirtualGenerator<T>> py_class(m, className.c_str());
    py_class.def(py::init<int, int>());
    py_class.def("build_submatrix", &Class::build_submatrix);
    py_class.def("nb_rows", &Class::nb_rows);
    py_class.def("nb_cols", &Class::nb_cols);
}

#endif