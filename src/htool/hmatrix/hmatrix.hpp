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

    // // Getters
    // py_class.def_property_readonly("shape", [](const Class &self) {
    //     return std::array<int, 2>{self.nb_rows(), self.nb_cols()};
    // });
    // py_class.def("get_perm_t", overload_cast_<>()(&Class::get_permt, py::const_));
    // py_class.def("get_perm_s", overload_cast_<>()(&Class::get_perms, py::const_));
    // py_class.def("get_MasterOffset_t", overload_cast_<>()(&Class::get_MasterOffset_t, py::const_));
    // py_class.def("get_MasterOffset_s", overload_cast_<>()(&Class::get_MasterOffset_s, py::const_));

    // // Linear algebra
    // py_class.def("__mul__", [](const Class &self, std::vector<T> b) {
    //     return self * b;
    // });
    // py_class.def("matvec", [](const Class &self, std::vector<T> b) {
    //     return self * b;
    // });
    // py_class.def("__matmul__", [](const Class &self, py::array_t<T, py::array::f_style | py::array::forcecast> B) {
    //     int mu;

    //     if (B.ndim() == 1) {
    //         mu = 1;
    //     } else if (B.ndim() == 2) {
    //         mu = B.shape()[1];
    //     } else {
    //         throw std::runtime_error("Wrong dimension for HMatrix-matrix product"); // LCOV_EXCL_LINE
    //     }
    //     if (B.shape()[0] != self.nb_cols()) {
    //         throw std::runtime_error("Wrong size for HMatrix-matrix product"); // LCOV_EXCL_LINE
    //     }

    //     std::vector<T> result(self.nb_rows() * mu, 0);

    //     self.mvprod_global_to_global(B.data(), result.data(), mu);

    //     if (B.ndim() == 1) {
    //         std::array<long int, 1> shape{self.nb_rows()};
    //         return py::array_t<T, py::array::f_style>(shape, result.data());
    //     } else {
    //         std::array<long int, 2> shape{self.nb_rows(), mu};
    //         return py::array_t<T, py::array::f_style>(shape, result.data());
    //     }
    // });

    py_class.def(
        "get_sub_hmatrix", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) {
            return &*hmatrix.get_sub_hmatrix(target_cluster, source_cluster);
        },
        py::return_value_policy::reference_internal);
    py_class.def("get_tree_parameters", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
        auto tree_parameters = htool::get_tree_parameters(hmatrix);
        return tree_parameters;
    });
    py_class.def("get_local_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
        auto information = htool::get_hmatrix_information(hmatrix);
        return information;
    });
    py_class.def("get_distributed_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, MPI_Comm_wrapper comm) {
        auto information = htool::get_distributed_hmatrix_information(hmatrix, comm);
        return information;
    });
}

#endif
