#ifndef HTOOL_HMATRIX_CPP
#define HTOOL_HMATRIX_CPP

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/utils/recompression.hpp>

#ifdef HAVE_MPI
#    include "../misc/wrapper_mpi.hpp"
#    include <htool/hmatrix/hmatrix_distributed_output.hpp>
#endif

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_HMatrix(py::module &m, const std::string &className) {

    using Class = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());
    py_class.def_property_readonly("shape", [](const Class &self) { return std::pair<int, int>(self.nb_rows(), self.nb_cols()); });
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
#ifdef HAVE_MPI
    py_class.def("get_distributed_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, MPI_Comm_wrapper comm) { return htool::get_distributed_hmatrix_information(hmatrix, comm); });
#endif
    py_class.def("get_target_cluster", &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster, py::return_value_policy::reference_internal);
    py_class.def("get_source_cluster", &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster, py::return_value_policy::reference_internal);

    m.def("recompression", &htool::recompression<CoefficientPrecision, CoordinatePrecision, std::function<void(LowRankMatrix<CoefficientPrecision> &)>>);
    m.def("recompression", [](HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) { recompression(hmatrix); });
    m.def("openmp_recompression", &htool::openmp_recompression<CoefficientPrecision, CoordinatePrecision, std::function<void(LowRankMatrix<CoefficientPrecision> &)>>);
    m.def("openmp_recompression", [](HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) { recompression(hmatrix); });

    py_class.def(
        "__mul__", [](const Class &self, const py::array_t<CoefficientPrecision, py::array::f_style> input) {
            if (input.ndim() != 1) {
                throw std::runtime_error("Wrong dimension for HMatrix-vector product"); // LCOV_EXCL_LINE
            }
            if (input.shape()[0] != self.get_source_cluster().get_size()) {
                throw std::runtime_error("Wrong size for HMatrix-vector product"); // LCOV_EXCL_LINE
            }
            py::array_t<CoefficientPrecision, py::array::f_style> result(self.get_target_cluster().get_size());
            std::fill_n(result.mutable_data(), self.get_target_cluster().get_size(), CoefficientPrecision(0));

            htool::Matrix<CoefficientPrecision> dense_mat(self.get_target_cluster().get_size(), self.get_source_cluster().get_size());
            copy_to_dense_in_user_numbering(self, dense_mat.data());
            char trans = 'N';
            htool::add_hmatrix_vector_product(trans, CoefficientPrecision(1), self, input.data(), CoefficientPrecision(0), result.mutable_data());

            return result;
        },
        "in"_a);
}

#endif
