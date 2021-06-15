#ifndef HTOOL_MATRIX_CPP
#define HTOOL_MATRIX_CPP

#include <htool/htool.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace htool;

template <typename T>
class IMatrixCpp : public IMatrix<T> {
  public:
    using IMatrix<T>::IMatrix;

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {

        py::array_t<T, py::array::f_style> mat(std::array<long int, 2>{M, N}, ptr, py::capsule(ptr));
        build_submatrix(std::vector<int>(rows, rows + M), std::vector<int>(cols, cols + N), mat);
    }

    virtual void build_submatrix(const std::vector<int> &J, const std::vector<int> &K, py::array_t<T, py::array::f_style> &mat) const {};
};

template <typename T>
class PyIMatrix : public IMatrixCpp<T> {
  public:
    using IMatrixCpp<T>::IMatrixCpp;
    // PyIMatrix(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_submatrix(const std::vector<int> &J, const std::vector<int> &K, py::array_t<T, py::array::f_style> &mat) const override {
        PYBIND11_OVERLOAD(
            void,            /* Return type */
            IMatrixCpp<T>,   /* Parent class */
            build_submatrix, /* Name of function in C++ (must match Python name) */
            J,
            K,
            mat /* Argument(s) */
        );
    }
};

template <typename T>
void declare_IMatrix(py::module &m, const std::string &className) {
    using Class = IMatrixCpp<T>;
    py::class_<Class, PyIMatrix<T>>(m, className.c_str())
        .def(py::init<int, int>())
        .def("build_submatrix", &Class::build_submatrix)
        .def("nb_rows", &Class::nb_rows)
        .def("nb_cols", &Class::nb_cols);
}

#endif