#ifndef HTOOL_HMATRIX_CPP
#define HTOOL_HMATRIX_CPP

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../misc/wrapper_mpi.hpp"

#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_distributed_output.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_HMatrix(py::module &m, const std::string &className) {

    using Class = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());

    py_class.def("to_dense", [](const Class &self) {
        std::array<long int, 2> shape{self.get_target_cluster().get_size(), self.get_source_cluster().get_size()};
        py::array_t<CoefficientPrecision, py::array::f_style> dense(shape);
        std::fill_n(dense.mutable_data(), dense.size(), CoefficientPrecision(0));
        copy_to_dense(self, dense.mutable_data());
        return dense;
    });

    py_class.def("to_dense_in_user_numbering", [](const Class &self) {
        std::array<long int, 2> shape{self.get_target_cluster().get_size(), self.get_source_cluster().get_size()};
        py::array_t<CoefficientPrecision, py::array::f_style> dense(shape);
        std::fill_n(dense.mutable_data(), dense.size(), CoefficientPrecision(0));
        copy_to_dense_in_user_numbering(self, dense.mutable_data());
        return dense;
    });

    py_class.def("__deepcopy__", [](const Class &self, py::dict) { return Class(self); }, "memo"_a);

    py_class.def("get_tree_parameters", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) { return htool::get_tree_parameters(hmatrix); });
    py_class.def("get_local_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) { return htool::get_hmatrix_information(hmatrix); });
    py_class.def("get_distributed_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, MPI_Comm_wrapper comm) { return htool::get_distributed_hmatrix_information(hmatrix, comm); });
}

#endif
