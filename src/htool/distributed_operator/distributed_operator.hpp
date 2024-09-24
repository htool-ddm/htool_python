#ifndef HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_HPP
#include "../misc/utility.hpp"
#include "../misc/wrapper_mpi.hpp"
#include <htool/distributed_operator/distributed_operator.hpp>
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_global_to_global.hpp>
#include <htool/distributed_operator/linalg/add_distributed_operator_vector_product_global_to_global.hpp>
#include <htool/distributed_operator/linalg/add_distributed_operator_vector_sub_product_global_to_local.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/matrix/matrix_view.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision>
void declare_distributed_operator(py::module &m, const std::string &class_name) {
    using Class = DistributedOperator<CoefficientPrecision>;

    py::class_<Class> py_class(m, class_name.c_str());
    py_class.def_property_readonly("shape", [](const Class &self) { return std::pair<int, int>(self.get_target_partition().get_global_size(), self.get_source_partition().get_global_size()); });
    py_class.def("add_global_to_local_operator", &Class::add_global_to_local_operator, py::keep_alive<1, 2>());
    py_class.def("add_local_to_local_operator", &Class::add_local_to_local_operator, py::keep_alive<1, 2>());

    // Linear algebra
    py_class.def(
        "__mul__", [](const Class &self, const py::array_t<CoefficientPrecision, py::array::f_style> input) {
            if (input.ndim() != 1) {
                throw std::runtime_error("Wrong dimension for DistributedOperator-vector product"); // LCOV_EXCL_LINE
            }
            if (input.shape()[0] != self.get_source_partition().get_global_size()) {
                throw std::runtime_error("Wrong size for DistributedOperator-vector product"); // LCOV_EXCL_LINE
            }
            py::array_t<CoefficientPrecision, py::array::f_style> result(std::array<long int, 1>{self.get_target_partition().get_global_size()});
            std::fill_n(result.mutable_data(), self.get_target_partition().get_global_size(), CoefficientPrecision(0));
            htool::add_distributed_operator_vector_product_global_to_global<CoefficientPrecision>('N', CoefficientPrecision(1), self, input.data(), CoefficientPrecision(0), result.mutable_data(), nullptr);

            return result;
        },
        "in"_a);

    py_class.def(
        "__matmul__", [](const Class &self, py::array_t<CoefficientPrecision, py::array::f_style> input) {
            int mu;
            if (input.ndim() == 2) {
                mu = input.shape()[1];
            } else {
                throw std::runtime_error("Wrong dimension for HMatrix-matrix product"); // LCOV_EXCL_LINE
            }
            if (input.shape()[0] != self.get_source_partition().get_global_size()) {
                throw std::runtime_error("Wrong size for HMatrix-matrix product"); // LCOV_EXCL_LINE
            }

            std::array<long int, 2> shape{self.get_target_partition().get_global_size(), mu};
            py::array_t<CoefficientPrecision, py::array::f_style> result(shape);
            std::fill_n(result.mutable_data(), self.get_target_partition().get_global_size() * mu, CoefficientPrecision(0));
            if (mu == 1) {
                htool::add_distributed_operator_vector_product_global_to_global<CoefficientPrecision>('N', CoefficientPrecision(1), self, input.data(), CoefficientPrecision(0), result.mutable_data(), nullptr);
                return result;
            }
            MatrixView<const CoefficientPrecision> input_view(self.get_source_partition().get_global_size(), mu, input.data());
            MatrixView<CoefficientPrecision> output_view(self.get_target_partition().get_global_size(), mu, result.mutable_data());
            CoefficientPrecision *work = nullptr;
            add_distributed_operator_matrix_product_global_to_global('N', CoefficientPrecision(1), self, input_view, CoefficientPrecision(0), output_view, work);

            return result;
        },
        py::arg("input").noconvert(true));

    py_class.def(
        "internal_sub_vector_product_global_to_local", [](const Class &self, const py::array_t<CoefficientPrecision, py::array::f_style> input, int offset) {
            if (input.ndim() != 1) {
                throw std::runtime_error("Wrong dimension for DistributedOperator-vector product"); // LCOV_EXCL_LINE
            }
            int rank;
            MPI_Comm_rank(self.get_comm(), &rank);
            py::array_t<CoefficientPrecision, py::array::f_style> result(std::array<long int, 1>{self.get_target_partition().get_size_of_partition(rank)});
            std::fill_n(result.mutable_data(), result.size(), CoefficientPrecision(0));
            htool::internal_add_distributed_operator_vector_sub_product_global_to_local<CoefficientPrecision>(self, input.data(), result.mutable_data(), 1, offset, input.shape()[0]);
            return result;
        });
}

#endif
