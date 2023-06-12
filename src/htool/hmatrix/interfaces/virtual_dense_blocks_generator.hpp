#ifndef HTOOL_DENSE_BLOCKS_GENERATOR_CPP
#define HTOOL_DENSE_BLOCKS_GENERATOR_CPP

#include <htool/hmatrix/interfaces/virtual_dense_blocks_generator.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class VirtualDenseBlocksGeneratorPython : public VirtualDenseBlocksGenerator<CoefficientPrecision> {

    const Cluster<CoordinatePrecision> &m_target_cluster;
    const Cluster<CoordinatePrecision> &m_source_cluster;

  public:
    using VirtualDenseBlocksGenerator<CoefficientPrecision>::VirtualDenseBlocksGenerator;

    VirtualDenseBlocksGeneratorPython(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) : VirtualDenseBlocksGenerator<CoefficientPrecision>(), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {}

    void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<int> &rows_offsets, const std::vector<int> &cols_offsets, std::vector<CoefficientPrecision *> &ptr) const override {
        int nb_blocks            = M.size();
        auto &target_permutation = m_target_cluster.get_permutation();
        auto &source_permutation = m_source_cluster.get_permutation();
        std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> vec_ptr;
        std::vector<py::array_t<int, py::array::f_style>> rows_ptr;
        std::vector<py::array_t<int, py::array::f_style>> cols_ptr;
        for (int i = 0; i < nb_blocks; i++) {
            rows_ptr.emplace_back(std::array<long int, 1>{M[i]}, target_permutation.data() + rows_offsets[i], py::capsule(target_permutation.data() + rows_offsets[i]));
            cols_ptr.emplace_back(std::array<long int, 1>{N[i]}, source_permutation.data() + cols_offsets[i], py::capsule(source_permutation.data() + cols_offsets[i]));
            vec_ptr.emplace_back(std::array<long int, 2>{M[i], N[i]}, ptr[i], py::capsule(ptr[i]));
        }

        build_dense_blocks(rows_ptr, cols_ptr, vec_ptr);
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_dense_blocks(const std::vector<py::array_t<int, py::array::f_style>> &rows, const std::vector<py::array_t<int, py::array::f_style>> &cols, std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> &blocks) const = 0; // LCOV_EXCL_LINE
};

template <typename CoefficientPrecision>
class PyVirtualDenseBlocksGenerator : public VirtualDenseBlocksGeneratorPython<CoefficientPrecision> {
  public:
    using VirtualDenseBlocksGeneratorPython<CoefficientPrecision>::VirtualDenseBlocksGeneratorPython;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void build_dense_blocks(const std::vector<py::array_t<int, py::array::f_style>> &rows, const std::vector<py::array_t<int, py::array::f_style>> &cols, std::vector<py::array_t<CoefficientPrecision, py::array::f_style>> &blocks) const override {
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

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void declare_custom_VirtualDenseBlocksGenerator(py::module &m, const std::string &className) {
    // using BaseClass = VirtualDenseBlocksGenerator<CoefficientPrecision>;
    // py::class_<BaseClass, std::shared_ptr<BaseClass>>(m, base_class_name.c_str());

    using Class = VirtualDenseBlocksGeneratorPython<CoefficientPrecision>;
    py::class_<Class, std::shared_ptr<Class>, PyVirtualDenseBlocksGenerator<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init<const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &>());
    py_class.def("build_dense_blocks", &Class::build_dense_blocks);
}

#endif
