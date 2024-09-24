#ifndef HTOOL_PYTHON_VIRTUAL_COARSE_SPACE_BUILDER_HPP
#define HTOOL_PYTHON_VIRTUAL_COARSE_SPACE_BUILDER_HPP

#include <htool/solvers/interfaces/virtual_coarse_space_builder.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename CoefficientPrecision, typename CoordinatePrecision = htool::underlying_type<CoefficientPrecision>>
class VirtualGeneoCoarseSpaceBuilderPython : public VirtualCoarseSpaceBuilder<CoefficientPrecision> {
    py::array_t<CoefficientPrecision, py::array::f_style> m_coarse_space;

    int m_size_wo_overlap;
    int m_size_with_overlap;
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &m_local_hmatrix;
    int m_geneo_nu                                                 = 2;
    htool::underlying_type<CoefficientPrecision> m_geneo_threshold = -1.;

  public:
    int get_geneo_nu() { return this->m_geneo_nu; }
    htool::underlying_type<CoefficientPrecision> get_geneo_threshold() { return this->m_geneo_threshold; }

    explicit VirtualGeneoCoarseSpaceBuilderPython(int size_wo_overlap, int size_with_overlap, const HMatrix<CoefficientPrecision> &Ai, int geneo_nu, htool::underlying_type<CoefficientPrecision> geneo_threshold) : m_size_wo_overlap(size_wo_overlap), m_size_with_overlap(size_with_overlap), m_local_hmatrix(Ai), m_geneo_nu(geneo_nu), m_geneo_threshold(geneo_threshold) {}

    Matrix<CoefficientPrecision> build_coarse_space() override {
        std::function<py::array_t<CoefficientPrecision>(py::array_t<CoefficientPrecision>)> hmatrix_callback;

        if (this->m_local_hmatrix.nb_cols() == this->m_size_with_overlap) {
            hmatrix_callback = [this](py::array_t<CoefficientPrecision> in) {
                std::fill(in.mutable_data() + this->m_size_wo_overlap, in.mutable_data() + this->m_size_with_overlap, CoefficientPrecision(0));
                py::array_t<CoefficientPrecision> out(this->m_local_hmatrix.nb_rows());
                std::fill_n(out.mutable_data(), this->m_local_hmatrix.nb_rows(), CoefficientPrecision(0));
                add_hmatrix_vector_product('N', CoefficientPrecision(1), this->m_local_hmatrix, in.data(), CoefficientPrecision(0), out.mutable_data());
                std::fill(out.mutable_data() + this->m_size_wo_overlap, out.mutable_data() + this->m_size_with_overlap, CoefficientPrecision(0));
                return out;
            };
        } else if (this->m_local_hmatrix.nb_cols() == this->m_size_wo_overlap) {
            hmatrix_callback = [this](py::array_t<CoefficientPrecision> in) {
                py::array_t<CoefficientPrecision> out(in.size());
                std::fill_n(out.mutable_data(), in.size(), CoefficientPrecision(0));
                internal_add_hmatrix_vector_product('N', CoefficientPrecision(1), this->m_local_hmatrix, in.data(), CoefficientPrecision(0), out.mutable_data());
                return out;
            };
        }

        compute_coarse_space(hmatrix_callback);
        Matrix<CoefficientPrecision> coarse_space_mat(m_coarse_space.shape()[0], m_coarse_space.shape()[1]);
        std::copy_n(m_coarse_space.data(), m_coarse_space.shape()[0] * m_coarse_space.shape()[1], coarse_space_mat.data()); // HPDDM deletes the coarse space, so we have to copy.
        return coarse_space_mat;
    }

    virtual void compute_coarse_space(std::function<py::array_t<CoefficientPrecision>(py::array_t<CoefficientPrecision>)> Ai) = 0;

    void set_coarse_space(py::array_t<CoefficientPrecision, py::array::f_style> coarse_space) {
        m_coarse_space = coarse_space;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class PyVirtualGeneoCoarseSpaceBuilder : public VirtualGeneoCoarseSpaceBuilderPython<CoefficientPrecision> {

  public:
    /* Inherit the constructors */
    using VirtualGeneoCoarseSpaceBuilderPython<CoefficientPrecision>::VirtualGeneoCoarseSpaceBuilderPython;

    /* Trampoline (need one for each virtual function) */
    void compute_coarse_space(std::function<py::array_t<CoefficientPrecision>(py::array_t<CoefficientPrecision>)> Ai) override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                       /* Return type */
            VirtualGeneoCoarseSpaceBuilderPython<CoefficientPrecision>, /* Parent class */
            compute_coarse_space,                                       /* Name of function in C++ (must match Python name) */
            Ai                                                          /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision>
void declare_virtual_coarse_space_builder(py::module &m, const std::string &className, const std::string &base_class_name) {
    using BaseClass = VirtualCoarseSpaceBuilder<CoefficientPrecision>;
    py::class_<BaseClass>(m, base_class_name.c_str());

    using Class = VirtualGeneoCoarseSpaceBuilderPython<CoefficientPrecision>;
    py::class_<Class, PyVirtualGeneoCoarseSpaceBuilder<CoefficientPrecision>, VirtualCoarseSpaceBuilder<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, const HMatrix<CoefficientPrecision> &Ai, int geneo_nu) {
                     return PyVirtualGeneoCoarseSpaceBuilder<CoefficientPrecision>(size_wo_overlap, size_with_overlap, Ai, geneo_nu, -1);
                 }),
                 py::arg("size_wo_overlap"), // LCOV_EXCL_START
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::kw_only(),
                 py::arg("geneo_nu"));
    // LCOV_EXCL_STOP
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, const HMatrix<CoefficientPrecision> &Ai, double geneo_threshold) {
                     return PyVirtualGeneoCoarseSpaceBuilder<CoefficientPrecision>(size_wo_overlap, size_with_overlap, Ai, 0, geneo_threshold);
                 }),
                 py::arg("size_wo_overlap"), // LCOV_EXCL_START
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::kw_only(),
                 py::arg("geneo_threshold"));
    // LCOV_EXCL_STOP
    py_class.def("set_coarse_space", &Class::set_coarse_space);
    py_class.def_property_readonly("geneo_nu", &Class::get_geneo_nu);
    py_class.def_property_readonly("geneo_threshold", &Class::get_geneo_threshold);
}
#endif
