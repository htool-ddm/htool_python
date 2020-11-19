
#include <htool/htool.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace htool;

template <typename T>
class PyIMatrix : public IMatrix<T> {
  public:
    using IMatrix<T>::IMatrix;
    // PyIMatrix(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    T get_coef(const int &i, const int &j) const override {
        PYBIND11_OVERLOAD_PURE(
            T,          /* Return type */
            IMatrix<T>, /* Parent class */
            get_coef,   /* Name of function in C++ (must match Python name) */
            i,
            j /* Argument(s) */
        );
    }

    SubMatrix<T> get_submatrix(const std::vector<int> &J, const std::vector<int> &K) const override {
        PYBIND11_OVERLOAD(
            SubMatrix<T>,  /* Return type */
            IMatrix<T>,    /* Parent class */
            get_submatrix, /* Name of function in C++ (must match Python name) */
            J,
            K /* Argument(s) */
        );
    }
};

template <typename T>
void declare_IMatrix(py::module &m, const std::string &className) {
    using Class = IMatrix<T>;
    py::class_<Class, PyIMatrix<T>>(m, className.c_str())
        .def(py::init<int, int>())
        .def("get_coef", &Class::get_coef)
        .def("get_submatrix", &Class::get_submatrix)
        .def("nb_rows", &Class::nb_rows)
        .def("nb_cols", &Class::nb_cols);
}

template <typename T>
class PySubMatrix : public SubMatrix<T> {
  public:
    using SubMatrix<T>::SubMatrix;
    PySubMatrix(const std::vector<int> &J, const std::vector<int> &K, py::array_t<T, py::array::f_style | py::array::forcecast> numpy_array) : SubMatrix<T>(J, K) {
        auto dims = numpy_array.ndim();
        if (dims != 2) {
            throw std::runtime_error("Wrong dimension for SubMatrix");
        }
        if (numpy_array.shape()[0] != J.size() || numpy_array.shape()[1] != K.size()) {
            throw std::runtime_error("Wrong size for SubMatrix");
        }
        std::move(numpy_array.data(), numpy_array.data() + J.size() * K.size(), this->mat.data());
        // No difference observed in terms of timing between move and copy_n
        // std::copy_n( numpy_array.data(), J.size()*K.size(), this->mat.data());
    }
};

template <typename T>
void declare_SubMatrix(py::module &m, const std::string &className) {
    using Class = SubMatrix<T>;
    py::class_<Class, PySubMatrix<T>>(m, className.c_str())
        .def(py::init_alias<const std::vector<int> &, const std::vector<int> &, py::array_t<T, py::array::f_style | py::array::forcecast>>())
        .def("get_matrix", [](const Class &self) {
            return py::array_t<T>({self.nb_rows(), self.nb_cols()}, self.get_mat().data());
        });
}
