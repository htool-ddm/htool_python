#ifndef HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_PYBIND11_DISTRIBUTED_OPERATOR_HPP
#include "../misc/utility.hpp"
#include "../misc/wrapper_mpi.hpp"
#include <htool/distributed_operator/distributed_operator.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision>
void declare_distributed_operator(py::module &m, const std::string &class_name) {
    using Class = DistributedOperator<CoefficientPrecision>;

    py::class_<Class> py_class(m, class_name.c_str());
    py_class.def(py::init<IPartition<CoefficientPrecision> &, IPartition<CoefficientPrecision> &, char, char, MPI_Comm_wrapper>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>());
    py_class.def("add_local_operator", &Class::add_local_operator, py::keep_alive<1, 2>());

    // Linear algebra
    py_class.def(
        "__mul__", [](const Class &self, const py::array_t<CoefficientPrecision, py::array::f_style> input) {
            if (input.ndim() != 1) {
                throw std::runtime_error("Wrong dimension for HMatrix-vector product"); // LCOV_EXCL_LINE
            }
            if (input.shape()[0] != self.get_source_partition().get_global_size()) {
                throw std::runtime_error("Wrong size for HMatrix-vector product"); // LCOV_EXCL_LINE
            }
            py::array_t<CoefficientPrecision, py::array::f_style> result(std::array<long int, 1>{self.get_target_partition().get_global_size()});
            self.vector_product_global_to_global(input.data(), result.mutable_data());

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

            self.matrix_product_global_to_global(input.data(), result.mutable_data(), mu);

            return result;
        },
        py::arg("input").noconvert(true));
}

#endif
