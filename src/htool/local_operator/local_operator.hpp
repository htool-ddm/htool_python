#ifndef HTOOL_LOCAL_OPERATOR_CPP
#define HTOOL_LOCAL_OPERATOR_CPP

#include <htool/distributed_operator/implementations/global_to_local_operators/restricted_operator.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision>
class RestrictedGlobalToLocalOperatorPython : public htool::RestrictedGlobalToLocalOperator<CoefficientPrecision> {
  public:
    using htool::RestrictedGlobalToLocalOperator<CoefficientPrecision>::RestrictedGlobalToLocalOperator;

    RestrictedGlobalToLocalOperatorPython(LocalRenumbering target_local_renumbering, LocalRenumbering source_local_renumbering, bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : RestrictedGlobalToLocalOperator<CoefficientPrecision>(target_local_renumbering, source_local_renumbering, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod) {}

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {

        py::array_t<CoefficientPrecision> input(std::array<long int, 1>{this->m_local_source_renumbering.get_size()}, in, py::capsule(in));
        py::array_t<CoefficientPrecision> output(std::array<long int, 1>{this->m_local_target_renumbering.get_size()}, out, py::capsule(out));

        add_vector_product(trans, alpha, input, beta, output);
    }

    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {

        py::array_t<CoefficientPrecision, py::array::c_style> input(std::array<long int, 2>{this->m_local_source_renumbering.get_size(), mu}, in, py::capsule(in));
        py::array_t<CoefficientPrecision, py::array::c_style> output(std::array<long int, 2>{this->m_local_target_renumbering.get_size(), mu}, out, py::capsule(out));

        add_matrix_product_row_major(trans, alpha, input, beta, output);
    }

    // lcov does not see it because of trampoline I assume
    virtual void add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const = 0; // LCOV_EXCL_LINE
    // virtual void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const std::vector<CoefficientPrecision> &in, CoefficientPrecision beta, std::vector<CoefficientPrecision> &out) const = 0; // LCOV_EXCL_LINE

    virtual void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const = 0; // LCOV_EXCL_LINE

    LocalRenumbering get_local_target_renumbering() const { return this->m_local_target_renumbering; }
    LocalRenumbering get_local_source_renumbering() const { return this->m_local_source_renumbering; }
};

template <typename CoefficientPrecision>
class PyRestrictedGlobalToLocalOperator : public RestrictedGlobalToLocalOperatorPython<CoefficientPrecision> {
  public:
    using RestrictedGlobalToLocalOperatorPython<CoefficientPrecision>::RestrictedGlobalToLocalOperatorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                        /* Return type */
            RestrictedGlobalToLocalOperatorPython<CoefficientPrecision>, /* Parent class */
            add_vector_product,                                          /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
    virtual void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                        /* Return type */
            RestrictedGlobalToLocalOperatorPython<CoefficientPrecision>, /* Parent class */
            add_matrix_product_row_major,                                /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision>
void declare_global_to_local_operator(py::module &m, const std::string &prefix) {
    using VirtualClass = htool::VirtualGlobalToLocalOperator<CoefficientPrecision>;
    py::class_<VirtualClass>(m, (prefix + "IGlobalToLocalOperator").c_str());

    using BaseClass = RestrictedGlobalToLocalOperator<CoefficientPrecision>;
    py::class_<BaseClass, VirtualClass> py_base_class(m, (prefix + "IRestrictedGlobalToLocalOperator").c_str());

    using Class = RestrictedGlobalToLocalOperatorPython<CoefficientPrecision>;
    py::class_<Class, PyRestrictedGlobalToLocalOperator<CoefficientPrecision>, BaseClass> py_class(m, (prefix + "RestrictedGlobalToLocalOperator").c_str());
    py_class.def(py::init<LocalRenumbering, LocalRenumbering, bool, bool>());
    py_class.def("add_vector_product", &Class::add_vector_product, py::arg("trans"), py::arg("alpha"), py::arg("in").noconvert(true), py::arg("beta"), py::arg("out").noconvert(true));
    py_class.def("add_matrix_product_row_major", &Class::add_matrix_product_row_major);
    py_class.def_property_readonly("local_target_renumbering", &Class::get_local_target_renumbering);
    py_class.def_property_readonly("local_source_renumbering", &Class::get_local_source_renumbering);
}

#endif
