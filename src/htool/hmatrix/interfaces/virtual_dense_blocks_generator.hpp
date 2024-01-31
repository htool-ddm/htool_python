#ifndef HTOOL_DENSE_BLOCKS_GENERATOR_CPP
#define HTOOL_DENSE_BLOCKS_GENERATOR_CPP

#include <htool/hmatrix/interfaces/virtual_dense_blocks_generator.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision>
class VirtualDenseBlocksGeneratorPython : public VirtualDenseBlocksGenerator<CoefficientPrecision> {

  public:
    using VirtualDenseBlocksGenerator<CoefficientPrecision>::VirtualDenseBlocksGenerator;

    void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<int> &rows_offsets, const std::vector<int> &cols_offsets, std::vector<CoefficientPrecision *> &ptr) const override {

        int nb_blocks = M.size();
        py::array_t<int> rows_offsets_np(rows_offsets.size(), rows_offsets.data(), py::capsule(rows_offsets.data()));
        py::array_t<int> cols_offsets_np(cols_offsets.size(), cols_offsets.data(), py::capsule(cols_offsets.data()));
        std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> vec_ptr;
        for (int i = 0; i < nb_blocks; i++) {
            vec_ptr.emplace_back(std::array<long int, 2>{M[i], N[i]}, ptr[i], py::capsule(ptr[i]));
        }

        build_dense_blocks(rows_offsets_np, cols_offsets_np, vec_ptr);
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_dense_blocks(const py::array_t<int> &rows_offsets_np, const py::array_t<int> &cols_offsets_np, std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> &blocks) const = 0; // LCOV_EXCL_LINE
};

template <typename CoefficientPrecision>
class PyVirtualDenseBlocksGenerator : public VirtualDenseBlocksGeneratorPython<CoefficientPrecision> {
  public:
    using VirtualDenseBlocksGeneratorPython<CoefficientPrecision>::VirtualDenseBlocksGeneratorPython;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_dense_blocks(const py::array_t<int> &rows, const py::array_t<int> &cols, std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> &blocks) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                    /* Return type */
            VirtualDenseBlocksGeneratorPython<CoefficientPrecision>, /* Parent class */
            build_dense_blocks,                                      /* Name of function in C++ (must match Python name) */
            rows,
            cols,
            blocks /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision>
void declare_custom_VirtualDenseBlocksGenerator(py::module &m, const std::string &className) {
    // using BaseClass = VirtualDenseBlocksGenerator<CoefficientPrecision>;
    // py::class_<BaseClass, std::shared_ptr<BaseClass>>(m, base_class_name.c_str());

    using Class = VirtualDenseBlocksGeneratorPython<CoefficientPrecision>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualDenseBlocksGenerator<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init<>());
    py_class.def("build_dense_blocks", &Class::build_dense_blocks);
}

#endif
