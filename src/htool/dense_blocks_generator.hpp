#ifndef HTOOL_DENSE_BLOCKS_GENERATOR_CPP
#define HTOOL_DENSE_BLOCKS_GENERATOR_CPP

#include <htool/htool.hpp>
#include <htool/types/virtual_dense_blocks_generator.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename T>
class VirtualDenseBlocksGeneratorCpp : public VirtualDenseBlocksGenerator<T> {

  public:
    using VirtualDenseBlocksGenerator<T>::VirtualDenseBlocksGenerator;

    void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<const int *> &rows, const std::vector<const int *> &cols, std::vector<T *> &ptr) const override {

        int nb_blocks = M.size();
        std::vector<py::array_t<T, py::array::f_style>> vec_ptr;
        std::vector<py::array_t<int, py::array::f_style>> rows_ptr;
        std::vector<py::array_t<int, py::array::f_style>> cols_ptr;
        for (int i = 0; i < nb_blocks; i++) {
            rows_ptr.emplace_back(std::array<long int, 1>{M[i]}, rows[i], py::capsule(rows[i]));
            cols_ptr.emplace_back(std::array<long int, 1>{N[i]}, cols[i], py::capsule(cols[i]));
            vec_ptr.emplace_back(std::array<long int, 2>{M[i], N[i]}, ptr[i], py::capsule(ptr[i]));
        }

        build_dense_blocks(rows_ptr, cols_ptr, vec_ptr);
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_dense_blocks(const std::vector<py::array_t<int, py::array::f_style>> &rows, const std::vector<py::array_t<int, py::array::f_style>> &cols, std::vector<py::array_t<T, py::array::f_style>> &blocks) const = 0; // LCOV_EXCL_LINE
};

template <typename T>
class PyVirtualDenseBlocksGenerator : public VirtualDenseBlocksGeneratorCpp<T> {
  public:
    using VirtualDenseBlocksGeneratorCpp<T>::VirtualDenseBlocksGeneratorCpp;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_dense_blocks(const std::vector<py::array_t<int, py::array::f_style>> &rows, const std::vector<py::array_t<int, py::array::f_style>> &cols, std::vector<py::array_t<T, py::array::f_style>> &blocks) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                              /* Return type */
            VirtualDenseBlocksGeneratorCpp<T>, /* Parent class */
            build_dense_blocks,                /* Name of function in C++ (must match Python name) */
            rows,
            cols,
            blocks /* Argument(s) */
        );
    }
};

template <typename T>
void declare_custom_VirtualDenseBlocksGenerator(py::module &m, const std::string &className) {
    using Class = VirtualDenseBlocksGeneratorCpp<T>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualDenseBlocksGenerator<T>> py_class(m, className.c_str());
    py_class.def(py::init<>());
    py_class.def("build_dense_blocks", &Class::build_dense_blocks);
}

#endif
