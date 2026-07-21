#ifndef HTOOL_GENERATOR_CPP
#define HTOOL_GENERATOR_CPP

#include <htool/hmatrix/interfaces/virtual_generator.hpp>
#include <pybind11/pybind11.h>

using namespace htool;

template <typename CoefficientPrecision>
class VirtualGeneratorPython : public htool::VirtualGenerator<CoefficientPrecision> {
  public:
    using VirtualGenerator<CoefficientPrecision>::VirtualGenerator;
    VirtualGeneratorPython(const py::array_t<int> &target_permutation, const py::array_t<int> &source_permutation) : VirtualGenerator<CoefficientPrecision>() {}

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, CoefficientPrecision *ptr) const override {
        if (M * N > 0) {
            py::gil_scoped_acquire acquire;
            py::array_t<CoefficientPrecision, py::array::f_style> mat(std::array<long int, 2>{M, N}, ptr, py::capsule(ptr));

            py::array_t<int> py_rows(std::array<long int, 1>{M}, rows, py::capsule(rows));
            py::array_t<int> py_cols(std::array<long int, 1>{N}, cols, py::capsule(cols));

            build_submatrix(py_rows, py_cols, mat);
        }
    }

    // lcov does not see it because of trampoline I assume
    virtual void build_submatrix(const py::array_t<int> &J, const py::array_t<int> &K, py::array_t<CoefficientPrecision, py::array::f_style> &mat) const = 0; // LCOV_EXCL_LINE
};

template <typename CoefficientPrecision>
class PyVirtualGenerator : public VirtualGeneratorPython<CoefficientPrecision> {
  public:
    using VirtualGeneratorPython<CoefficientPrecision>::VirtualGeneratorPython;

    /* Trampoline (need one for each virtual function) */
    virtual void build_submatrix(const py::array_t<int> &J, const py::array_t<int> &K, py::array_t<CoefficientPrecision, py::array::f_style> &mat) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                         /* Return type */
            VirtualGeneratorPython<CoefficientPrecision>, /* Parent class */
            build_submatrix,                              /* Name of function in C++ (must match Python name) */
            J,
            K,
            mat /* Argument(s) */
        );
    }
};

template <typename CoefficientPrecision>
class GeneratorFromCapsule : public htool::VirtualGenerator<CoefficientPrecision> {
  private:
    using backend_callback_t = void (*)(int, int, const int *, const int *, double *, void *);

    struct Backend {
        backend_callback_t callback;
        void *ctx;
    };

    backend_callback_t callback = nullptr;
    void *ctx                   = nullptr;

    py::capsule m_capsule;

  public:
    using VirtualGenerator<CoefficientPrecision>::VirtualGenerator;
    GeneratorFromCapsule(py::object obj) : VirtualGenerator<CoefficientPrecision>() {
        if (!py::isinstance<py::capsule>(obj)) {
            throw std::runtime_error("Backend needs to be a Python capsule");
        }

        m_capsule = obj.cast<py::capsule>();

        auto *user_backend = static_cast<Backend *>(
            PyCapsule_GetPointer(m_capsule.ptr(), "backend"));

        if (!user_backend || !user_backend->callback) {
            throw std::runtime_error("Invalid backend capsule");
        }

        callback = user_backend->callback;
        ctx      = user_backend->ctx;
    }

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, CoefficientPrecision *ptr) const override {
        double *user_ptr = reinterpret_cast<double *>(ptr);
        if (M * N > 0) {
            callback(M, N, rows, cols, user_ptr, ctx);
        }
    }
};

template <typename CoefficientPrecision>
void declare_virtual_generator(py::module &m, const std::string &prefix) {
    using BaseClass = VirtualGenerator<CoefficientPrecision>;
    py::class_<BaseClass>(m, (prefix + "IGenerator").c_str());

    using Class = VirtualGeneratorPython<CoefficientPrecision>;
    py::class_<Class, BaseClass, PyVirtualGenerator<CoefficientPrecision>> py_class(m, (prefix + "VirtualGenerator").c_str());
    py_class.def(py::init<>());
    py_class.def("build_submatrix", &Class::build_submatrix);

    py::class_<GeneratorFromCapsule<CoefficientPrecision>, BaseClass> py_class_2(m, (prefix + "GeneratorFromCapsule").c_str());
    py_class_2.def(py::init<py::object>());
}

#endif
