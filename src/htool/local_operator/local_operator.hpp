#ifndef HTOOL_LOCAL_OPERATOR_CPP
#define HTOOL_LOCAL_OPERATOR_CPP

#include <htool/local_operators/local_operator.hpp>
#include <htool/local_operators/virtual_local_operator.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class LocalOperatorPython : public htool::LocalOperator<CoefficientPrecision, CoordinatePrecision> {
  public:
    using htool::LocalOperator<CoefficientPrecision, CoordinatePrecision>::LocalOperator;

    LocalOperatorPython(const Cluster<CoordinatePrecision> &cluster_tree_target, const Cluster<CoordinatePrecision> &cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod) {}

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {

        py::array_t<CoefficientPrecision> input(this->m_source_cluster.get_size(), in, py::capsule(in));
        py::array_t<CoefficientPrecision> output(this->m_target_cluster.get_size(), out, py::capsule(out));

        add_vector_product(trans, alpha, input, beta, output);
    }

    void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char UPLO, char symmetry) const override {

        py::array_t<CoefficientPrecision> input(this->m_source_cluster.get_size(), in, py::capsule(in));
        py::array_t<CoefficientPrecision> output(this->m_target_cluster.get_size(), out, py::capsule(out));

        add_vector_product(trans, alpha, input, beta, output);
    }

    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {

        py::array_t<CoefficientPrecision, py::array::c_style> input(std::array<long int, 2>{this->m_source_cluster.get_size(), mu}, in, py::capsule(in));
        py::array_t<CoefficientPrecision, py::array::c_style> output(std::array<long int, 2>{this->m_target_cluster.get_size(), mu}, out, py::capsule(out));

        add_matrix_product_row_major(trans, alpha, input, beta, output);
    }

    void local_add_matrix_product_symmetric_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char UPLO, char symmetry) const override {

        py::array_t<CoefficientPrecision, py::array::c_style> input(std::array<long int, 2>{this->m_source_cluster.get_size(), 1}, in, py::capsule(in));
        py::array_t<CoefficientPrecision, py::array::c_style> output(std::array<long int, 2>{this->m_target_cluster.get_size(), 1}, out, py::capsule(out));

        add_matrix_product_row_major(trans, alpha, input, beta, output);
    }

    // lcov does not see it because of trampoline I assume
    virtual void add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const = 0; // LCOV_EXCL_LINE
    // virtual void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const std::vector<CoefficientPrecision> &in, CoefficientPrecision beta, std::vector<CoefficientPrecision> &out) const = 0; // LCOV_EXCL_LINE

    virtual void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const = 0; // LCOV_EXCL_LINE
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
class PyLocalOperator : public LocalOperatorPython<CoefficientPrecision, CoordinatePrecision> {
  public:
    using LocalOperatorPython<CoefficientPrecision, CoordinatePrecision>::LocalOperatorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                      /* Return type */
            LocalOperatorPython<CoefficientPrecision>, /* Parent class */
            add_vector_product,                        /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
    virtual void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                      /* Return type */
            LocalOperatorPython<CoefficientPrecision>, /* Parent class */
            add_matrix_product_row_major,              /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_local_operator(py::module &m, const std::string &class_name) {
    using VirtualClass = htool::VirtualLocalOperator<CoefficientPrecision>;
    py::class_<VirtualClass>(m, ("Virtual" + class_name).c_str());

    using BaseClass = LocalOperator<CoefficientPrecision, CoordinatePrecision>;
    py::class_<BaseClass, VirtualClass> py_base_class(m, ("Base" + class_name).c_str());

    using Class = LocalOperatorPython<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class, PyLocalOperator<CoefficientPrecision, CoordinatePrecision>, BaseClass> py_class(m, class_name.c_str());
    py_class.def(py::init<const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, char, char, bool, bool>());
    py_class.def("add_vector_product", &Class::add_vector_product, py::arg("trans"), py::arg("alpha"), py::arg("in").noconvert(true), py::arg("beta"), py::arg("out").noconvert(true));
    py_class.def("add_matrix_product_row_major", &Class::add_matrix_product_row_major);
}

#endif
