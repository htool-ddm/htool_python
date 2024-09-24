#ifndef HTOOL_VIRTUAL_LOCAL_OPERATOR_CPP
#define HTOOL_VIRTUAL_LOCAL_OPERATOR_CPP

#include <htool/distributed_operator/interfaces/virtual_local_to_local_operator.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision = htool::underlying_type<CoefficientPrecision>>
class VirtualLocalToLocalOperatorPython : public htool::VirtualLocalToLocalOperator<CoefficientPrecision> {
    LocalRenumbering m_local_target_renumbering;
    LocalRenumbering m_local_source_renumbering;

  public:
    VirtualLocalToLocalOperatorPython(LocalRenumbering local_target_renumbering, LocalRenumbering local_source_renumbering) : m_local_target_renumbering(local_target_renumbering), m_local_source_renumbering(local_source_renumbering) {}

    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out) const override {
        py::array_t<CoefficientPrecision> input(std::array<long int, 1>{trans == 'N' ? m_local_source_renumbering.get_size() : m_local_target_renumbering.get_size()}, in, py::capsule(in));
        py::array_t<CoefficientPrecision> output(std::array<long int, 1>{trans == 'N' ? m_local_target_renumbering.get_size() : m_local_source_renumbering.get_size()}, out, py::capsule(out));

        local_add_vector_product(trans, alpha, input, beta, output);
    }
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out, int mu) const override {
        py::array_t<CoefficientPrecision, py::array::c_style> input(std::array<long int, 2>{trans == 'N' ? m_local_source_renumbering.get_size() : m_local_target_renumbering.get_size(), mu}, in, py::capsule(in));
        py::array_t<CoefficientPrecision, py::array::c_style> output(std::array<long int, 2>{trans == 'N' ? m_local_target_renumbering.get_size() : m_local_source_renumbering.get_size(), mu}, out, py::capsule(out));

        local_add_matrix_product_row_major(trans, alpha, input, beta, output);
    }

    void add_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const override {
        int source_offset = m_local_source_renumbering.get_offset();
        int source_size   = m_local_source_renumbering.get_size();

        int source_end = source_size + source_offset;
        int end        = size + offset;

        int temp_offset = std::max(offset, source_offset);
        int temp_end    = std::min(source_end, end);

        bool is_output_null = temp_end - temp_offset <= 0 ? true : false;
        if (offset == source_offset && temp_end == source_end) {
            add_matrix_product_row_major('N', 1, in, 1, out, mu);
        } else {
            std::vector<CoefficientPrecision> extension_by_zero(source_size * mu, 0);
            if (!is_output_null) {
                const CoefficientPrecision *const temp_in = in + temp_offset - offset;
                int temp_size                             = temp_end - temp_offset;
                std::copy_n(temp_in, temp_size * mu, extension_by_zero.data() + (temp_offset - source_offset) * mu);
            }
            add_matrix_product_row_major('N', 1, extension_by_zero.data(), 1, out, mu);
        }
    }

    virtual void local_add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const = 0; // LCOV_EXCL_LINE

    virtual void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const = 0; // LCOV_EXCL_LINE
};

template <typename CoefficientPrecision>
class PyVirtualLocalToLocalOperator : public VirtualLocalToLocalOperatorPython<CoefficientPrecision> {
  public:
    using VirtualLocalToLocalOperatorPython<CoefficientPrecision>::VirtualLocalToLocalOperatorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void local_add_vector_product(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                    /* Return type */
            VirtualLocalToLocalOperatorPython<CoefficientPrecision>, /* Parent class */
            local_add_vector_product,                                /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
    virtual void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const py::array_t<CoefficientPrecision, py::array::c_style> &in, CoefficientPrecision beta, py::array_t<CoefficientPrecision, py::array::c_style> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                    /* Return type */
            VirtualLocalToLocalOperatorPython<CoefficientPrecision>, /* Parent class */
            local_add_matrix_product_row_major,                      /* Name of function in C++ (must match Python name) */
            trans,
            alpha,
            in,
            beta,
            out /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = htool::underlying_type<CoefficientPrecision>>
void declare_virtual_local_to_local_operator(py::module &m, const std::string &prefix) {
    using BaseClass = htool::VirtualLocalToLocalOperator<CoefficientPrecision>;
    py::class_<BaseClass>(m, (prefix + "ILocalToLocalOperator").c_str());

    using Class = VirtualLocalToLocalOperatorPython<CoefficientPrecision>;
    py::class_<Class, BaseClass, PyVirtualLocalToLocalOperator<CoefficientPrecision>> py_class(m, (prefix + "VirtualLocalToLocalOperator").c_str());
    py_class.def(py::init<LocalRenumbering, LocalRenumbering>());
    py_class.def("local_add_vector_product", &Class::local_add_vector_product, py::arg("trans"), py::arg("alpha"), py::arg("in").noconvert(true), py::arg("beta"), py::arg("out").noconvert(true));
    py_class.def("local_add_matrix_product_row_major", &Class::local_add_matrix_product_row_major);
}

#endif
