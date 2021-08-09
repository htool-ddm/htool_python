
#ifndef HTOOL_WRAPPER_MPI_CPP
#define HTOOL_WRAPPER_MPI_CPP

// https://stackoverflow.com/questions/49259704/pybind11-possible-to-use-mpi4py
// https://stackoverflow.com/questions/52657173/sharing-an-mpi-communicator-using-pybind11
// https://bitbucket.org/fenics-project/dolfin/commits/025c96c331ce#Lpython/src/common.cppT139

#include <mpi.h>
#include <mpi4py/mpi4py.h>

// Issue with OpenMPI, they use void* for MPI_Comm
struct ompi_communicator_t {};

struct MPI_Comm_wrapper {
    MPI_Comm_wrapper() = default;
    MPI_Comm_wrapper(MPI_Comm value) : value(value) {}
    operator MPI_Comm() { return value; }
    MPI_Comm_wrapper &operator=(const MPI_Comm comm) {
        this->value = comm;
        return *this;
    }
    MPI_Comm value;
};

namespace pybind11 {
namespace detail {
template <>
struct type_caster<MPI_Comm_wrapper> {
  public:
    PYBIND11_TYPE_CASTER(MPI_Comm_wrapper, _("MPI_Comm_wrapper"));

    // Python -> C++
    bool load(handle src, bool) {
        PyObject *py_src = src.ptr();

        // Check that we have been passed an mpi4py communicator
        if (PyObject_TypeCheck(py_src, &PyMPIComm_Type)) {
            // Convert to regular MPI communicator
            value.value = *PyMPIComm_Get(py_src);
        } else {
            return false; // LCOV_EXCL_LINE
        }

        return !PyErr_Occurred();
    }

    // C++ -> Python
    static handle cast(MPI_Comm_wrapper src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        // Create an mpi4py handle
        return PyMPIComm_New(src.value);
    }
};
} // namespace detail
} // namespace pybind11
#endif