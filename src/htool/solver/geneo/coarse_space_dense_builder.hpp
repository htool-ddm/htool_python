#ifndef HTOOL_PYTHON_GENEO_COARSE_SPACE_DENSE_BUILDER_HPP
#define HTOOL_PYTHON_GENEO_COARSE_SPACE_DENSE_BUILDER_HPP

#include <htool/solvers/geneo/coarse_space_builder.hpp>
#include <pybind11/pybind11.h>
namespace py = pybind11;

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class GeneoCoarseSpaceDenseBuilderPython : public GeneoCoarseSpaceDenseBuilder<CoefficientPrecision> {
    py::array_t<CoefficientPrecision, py::array::f_style> m_coarse_space;

    // using GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>::GeneoCoarseSpaceDenseBuilder;
  public:
    char get_symmetry() { return this->m_symmetry; }
    int get_geneo_nu() { return this->m_geneo_nu; }
    htool::underlying_type<CoefficientPrecision> get_geneo_threshold() { return this->m_geneo_threshold; }

    explicit GeneoCoarseSpaceDenseBuilderPython(int size_wo_overlap, int size_with_overlap, const Matrix<CoefficientPrecision> &Ai, const Matrix<CoefficientPrecision> &Bi, char symmetry, char uplo, int geneo_nu, htool::underlying_type<CoefficientPrecision> geneo_threshold) : GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>(size_wo_overlap, size_with_overlap, Ai, Bi, symmetry, uplo, geneo_nu, geneo_threshold) {}

    Matrix<CoefficientPrecision> build_coarse_space() override {
        py::array_t<CoefficientPrecision, py::array::f_style> Ai(std::array<long int, 2>{this->m_DAiD.nb_rows(), this->m_DAiD.nb_cols()}, this->m_DAiD.data(), py::capsule(this->m_DAiD.data()));
        py::array_t<CoefficientPrecision, py::array::f_style> Bi(std::array<long int, 2>{this->m_Bi.nb_rows(), this->m_Bi.nb_cols()}, this->m_Bi.data(), py::capsule(this->m_Bi.data()));
        compute_coarse_space(Ai, Bi);
        Matrix<CoefficientPrecision> coarse_space_mat(m_coarse_space.shape()[0], m_coarse_space.shape()[1]);
        std::copy_n(m_coarse_space.data(), m_coarse_space.shape()[0] * m_coarse_space.shape()[1], coarse_space_mat.data()); // HPDDM deletes the coarse space, so we have to copy.
        // coarse_space_mat.assign(m_coarse_space.shape()[0], m_coarse_space.shape()[1], m_coarse_space.mutable_data(), false);
        // std::cout << "ICI?\n";
        // coarse_space_mat.print(std::cout, ",");
        return coarse_space_mat;
    }

    virtual void compute_coarse_space(py::array_t<CoefficientPrecision, py::array::f_style> Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi) = 0;

    void set_coarse_space(py::array_t<CoefficientPrecision, py::array::f_style> coarse_space) {
        m_coarse_space = coarse_space;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class PyGeneoCoarseSpaceDenseBuilder : public GeneoCoarseSpaceDenseBuilderPython<CoefficientPrecision> {

  public:
    /* Inherit the constructors */
    using GeneoCoarseSpaceDenseBuilderPython<CoefficientPrecision>::GeneoCoarseSpaceDenseBuilderPython;

    // explicit PyGeneoCoarseSpaceDenseBuilder(int size_wo_overlap, Matrix<CoefficientPrecision> Ai, Matrix<CoefficientPrecision> Bi, char symmetry, char uplo, int geneo_nu, htool::underlying_type<CoefficientPrecision> geneo_threshold) : GeneoCoarseSpaceDenseBuilderPython<CoefficientPrecision>(size_wo_overlap, Ai, Bi, symmetry, uplo, geneo_nu, geneo_threshold) {}

    /* Trampoline (need one for each virtual function) */
    void compute_coarse_space(py::array_t<CoefficientPrecision, py::array::f_style> Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi) override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                     /* Return type */
            GeneoCoarseSpaceDenseBuilderPython<CoefficientPrecision>, /* Parent class */
            compute_coarse_space,                                     /* Name of function in C++ (must match Python name) */
            Ai,
            Bi /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision>
void declare_geneo_coarse_space_dense_builder(py::module &m, const std::string &className) {

    using Class = GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>;
    py::class_<Class, VirtualCoarseSpaceBuilder<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, const HMatrix<CoefficientPrecision, underlying_type<CoefficientPrecision>> &Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi, char symmetry, char uplo, int geneo_nu) {
                     Matrix<CoefficientPrecision> Bi_mat;
                     Bi_mat.assign(Bi.shape()[0], Bi.shape()[1], Bi.mutable_data(), false);
                     return Class::GeneoWithNu(size_wo_overlap, size_with_overlap, Ai, Bi_mat, symmetry, uplo, geneo_nu);
                 }),
                 py::arg("size_wo_overlap"),
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::arg("Bi"),
                 py::arg("symmetry"),
                 py::arg("uplo"),
                 py::kw_only(),
                 py::arg("geneo_nu"));
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, const HMatrix<CoefficientPrecision, underlying_type<CoefficientPrecision>> &Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi, char symmetry, char uplo, double geneo_threshold) {
                     Matrix<CoefficientPrecision> Bi_mat;
                     Bi_mat.assign(Bi.shape()[0], Bi.shape()[1], Bi.mutable_data(), false);
                     return Class::GeneoWithThreshold(size_wo_overlap, size_with_overlap, Ai, Bi_mat, symmetry, uplo, geneo_threshold);
                 }),
                 py::arg("size_wo_overlap"),
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::arg("Bi"),
                 py::arg("symmetry"),
                 py::arg("uplo"),
                 py::kw_only(),
                 py::arg("geneo_threshold"));
}

template <typename CoefficientPrecision>
void declare_virtual_geneo_coarse_space_dense_builder(py::module &m, const std::string &className) {

    using Class = GeneoCoarseSpaceDenseBuilderPython<CoefficientPrecision>;
    py::class_<Class, PyGeneoCoarseSpaceDenseBuilder<CoefficientPrecision>, VirtualCoarseSpaceBuilder<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, py::array_t<CoefficientPrecision, py::array::f_style> Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi, char symmetry, char uplo, int geneo_nu) {
                     Matrix<CoefficientPrecision> Ai_mat;
                     Ai_mat.assign(Ai.shape()[0], Ai.shape()[1], Ai.mutable_data(), false);
                     Matrix<CoefficientPrecision> Bi_mat;
                     Bi_mat.assign(Bi.shape()[0], Bi.shape()[1], Bi.mutable_data(), false);
                     return PyGeneoCoarseSpaceDenseBuilder<CoefficientPrecision>(size_wo_overlap, size_with_overlap, Ai_mat, Bi_mat, symmetry, uplo, geneo_nu, -1);
                 }),
                 py::arg("size_wo_overlap"),
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::arg("Bi"),
                 py::arg("symmetry"),
                 py::arg("uplo"),
                 py::kw_only(),
                 py::arg("geneo_nu"));
    py_class.def(py::init([](int size_wo_overlap, int size_with_overlap, py::array_t<CoefficientPrecision, py::array::f_style> Ai, py::array_t<CoefficientPrecision, py::array::f_style> Bi, char symmetry, char uplo, double geneo_threshold) {
                     Matrix<CoefficientPrecision> Ai_mat;
                     Ai_mat.assign(Ai.shape()[0], Ai.shape()[1], Ai.mutable_data(), false);
                     Matrix<CoefficientPrecision> Bi_mat;
                     Bi_mat.assign(Bi.shape()[0], Bi.shape()[1], Bi.mutable_data(), false);
                     return PyGeneoCoarseSpaceDenseBuilder<CoefficientPrecision>(size_wo_overlap, size_with_overlap, Ai_mat, Bi_mat, symmetry, uplo, 0, geneo_threshold);
                 }),
                 py::arg("size_wo_overlap"),
                 py::arg("size_with_overlap"),
                 py::arg("Ai"),
                 py::arg("Bi"),
                 py::arg("symmetry"),
                 py::arg("uplo"),
                 py::kw_only(),
                 py::arg("geneo_threshold"));
    py_class.def("set_coarse_space", &Class::set_coarse_space);
    py_class.def_property_readonly("symmetry", &Class::get_symmetry);
    py_class.def_property_readonly("geneo_nu", &Class::get_geneo_nu);
    py_class.def_property_readonly("geneo_threshold", &Class::get_geneo_threshold);
}
#endif
