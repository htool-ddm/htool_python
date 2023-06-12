#ifndef HTOOL_PYTHON_VIRTUAL_COARSE_SPACE_BUILDER_HPP
#define HTOOL_PYTHON_VIRTUAL_COARSE_SPACE_BUILDER_HPP

#include <htool/solvers/interfaces/virtual_coarse_space_builder.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename CoefficientPrecision>
class VirtualCoarseSpaceBuilderPython : public VirtualCoarseSpaceBuilder<CoefficientPrecision> {
    py::array_t<CoefficientPrecision, py::array::f_style> m_coarse_space;

  public:
    Matrix<CoefficientPrecision> build_coarse_space() {
        compute_coarse_space();
        if (m_coarse_space.ndim() != 2) {
            htool::Logger::get_instance()
                .log(LogLevel::ERROR, "Wrong dimension for coarse space matrix when building coarse space."); // LCOV_EXCL_LINE
        }

        Matrix<CoefficientPrecision> coarse_space(m_coarse_space.shape()[0], m_coarse_space.shape()[1]);
        std::copy_n(m_coarse_space.data(), m_coarse_space.shape()[0] * m_coarse_space.shape()[1], coarse_space.data());
        return coarse_space;
    }

    // lcov does not see it because of trampoline I assume
    virtual void compute_coarse_space() const = 0; // LCOV_EXCL_LINE

    void set_coarse_space(py::array_t<CoefficientPrecision, py::array::f_style> coarse_space) {
        m_coarse_space = coarse_space;
    }
};

template <typename CoefficientPrecision>
class PyVirtualCoarseSpaceBuilder : public VirtualCoarseSpaceBuilderPython<CoefficientPrecision> {
  public:
    using VirtualCoarseSpaceBuilderPython<CoefficientPrecision>::VirtualCoarseSpaceBuilderPython;

    /* Trampoline (need one for each virtual function) */
    virtual void compute_coarse_space() const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                                  /* Return type */
            VirtualCoarseSpaceBuilderPython<CoefficientPrecision>, /* Parent class */
            compute_coarse_space                                   /* Name of function in C++ (must match Python name) */
        );
    }
};

template <typename CoefficientPrecision>
void declare_virtual_coarse_space_builder(py::module &m, const std::string &className, const std::string &base_class_name) {
    using BaseClass = VirtualCoarseSpaceBuilder<CoefficientPrecision>;
    py::class_<BaseClass>(m, base_class_name.c_str());

    using Class = VirtualCoarseSpaceBuilderPython<CoefficientPrecision>;
    py::class_<Class, BaseClass, PyVirtualCoarseSpaceBuilder<CoefficientPrecision>> py_class(m, className.c_str());
    py_class.def(py::init<>());
    py_class.def("compute_coarse_space", &Class::compute_coarse_space);
    py_class.def("set_coarse_space", &Class::set_coarse_space);
}
#endif
