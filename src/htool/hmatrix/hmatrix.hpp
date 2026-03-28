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
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/utils/recompression.hpp>
#include <htool/matrix/matrix_view.hpp>

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

    py_class.def("lu_factorization", [](HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
        htool::lu_factorization(hmatrix);
    });
    py_class.def("cholesky_factorization", [](HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, char UPLO) {
        htool::cholesky_factorization(UPLO, hmatrix);
    });
    py_class.def("lu_solve", [](const Class &self, char trans, const py::array_t<CoefficientPrecision, py::array::f_style> &input) {
        std::vector<ssize_t> shape;
        if (input.ndim() == 1) {
            shape = {input.shape()[0]};
        } else if (input.ndim() == 2) {
            shape = {input.shape()[0], input.shape()[1]};
        } else {
            throw std::runtime_error("Wrong dimension for HMatrix-LU input"); // LCOV_EXCL_LINE
        }
        py::array_t<CoefficientPrecision, py::array::f_style> result(shape);
        std::copy_n(input.data(), input.size(), result.mutable_data());
        htool::MatrixView<CoefficientPrecision> output_view(result.shape()[0], input.ndim() == 1 ? 1 : result.shape()[1], result.mutable_data());
        htool::lu_solve(trans, self, output_view);
        return result;
    });

    py_class.def("cholesky_solve", [](const Class &self, char UPLO, const py::array_t<CoefficientPrecision, py::array::f_style> &input) {
        std::vector<ssize_t> shape;
        if (input.ndim() == 1) {
            shape = {input.shape()[0]};
        } else if (input.ndim() == 2) {
            shape = {input.shape()[0], input.shape()[1]};
        } else {
            throw std::runtime_error("Wrong dimension for HMatrix-Cholesky input"); // LCOV_EXCL_LINE
        }
        py::array_t<CoefficientPrecision, py::array::f_style> result(shape);
        std::copy_n(input.data(), input.size(), result.mutable_data());
        htool::MatrixView<CoefficientPrecision> output_view(result.shape()[0], input.ndim() == 1 ? 1 : result.shape()[1], result.mutable_data());
        htool::cholesky_solve(UPLO, self, output_view);
        return result;
    });

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

            char trans = 'N';
            htool::add_hmatrix_vector_product(trans, CoefficientPrecision(1), self, input.data(), CoefficientPrecision(0), result.mutable_data());

            return result;
        },
        "in"_a);

    py_class.def(
        "__matmul__", [](const Class &self, const py::array_t<CoefficientPrecision, py::array::f_style> input) {
            if (input.ndim() != 2) {
                throw std::runtime_error("Wrong dimension for HMatrix-matrix product"); // LCOV_EXCL_LINE
            }
            if (input.shape()[0] != self.get_source_cluster().get_size()) {
                throw std::runtime_error("Wrong size for HMatrix-matrix product"); // LCOV_EXCL_LINE
            }
            py::array_t<CoefficientPrecision, py::array::f_style> result({input.shape()[0], input.shape()[1]});
            std::fill_n(result.mutable_data(), input.shape()[0] * input.shape()[1], CoefficientPrecision(0));

            htool::MatrixView<const CoefficientPrecision> input_view(input.shape()[0], input.shape()[1], input.data());
            htool::MatrixView<CoefficientPrecision> output_view(input.shape()[0], input.shape()[1], result.mutable_data());
            char transa = 'N';
            char transb = 'N';
            htool::add_hmatrix_matrix_product(transa, transb, CoefficientPrecision(1), self, input_view, CoefficientPrecision(0), output_view);

            return result;
        },
        "in"_a);
}

#endif
