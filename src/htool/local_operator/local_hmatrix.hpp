#ifndef HTOOL_LOCAL_HMATRIX_CPP
#define HTOOL_LOCAL_HMATRIX_CPP

#include "../misc/wrapper_mpi.hpp"
#include "local_operator.hpp"
#include <htool/hmatrix/hmatrix_distributed_output.hpp>
#include <htool/local_operators/local_hmatrix.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_local_hmatrix(py::module &m, const std::string &class_name) {
    using Class = htool::LocalHMatrix<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class, htool::LocalOperator<CoefficientPrecision, CoordinatePrecision>> py_data(m, class_name.c_str());

    py_data.def(py::init<const HMatrix<CoefficientPrecision, CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, const Cluster<CoordinatePrecision> &, char, char, bool, bool>(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>(), py::keep_alive<1, 4>());

    // py_data.def("print_information", [](const Class &self, MPI_Comm_wrapper comm) {
    //     htool::print_distributed_hmatrix_information(self.get_hmatrix(), std::cout, comm);
    // });

    // py_data.def("display", [](const Class &self) {
    //     std::vector<DisplayBlock<int>> output_blocks{};
    //     const HMatrix<CoefficientPrecision> &hmatrix = self.get_hmatrix();
    //     preorder_tree_traversal(
    //         hmatrix,
    //         [&output_blocks, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatePrecision> &current_hmatrix) {
    //             if (current_hmatrix.is_leaf()) {
    //                 output_blocks.push_back(DisplayBlock<int>{current_hmatrix.get_target_cluster().get_offset() - hmatrix.get_target_cluster().get_offset(), current_hmatrix.get_source_cluster().get_offset() - hmatrix.get_source_cluster().get_offset(), current_hmatrix.get_target_cluster().get_size(), current_hmatrix.get_source_cluster().get_size(), current_hmatrix.get_rank()});
    //             }
    //         });

    //     // Import
    //     py::object plt     = py::module::import("matplotlib.pyplot");
    //     py::object patches = py::module::import("matplotlib.patches");
    //     py::object colors  = py::module::import("matplotlib.colors");
    //     py::object numpy   = py::module::import("numpy");

    //     // First Data
    //     int nr = self.nb_rows();
    //     int nc = self.nb_cols();
    //     py::array_t<int> matrix({nr, nc});
    //     py::array_t<bool> mask_matrix({nr, nc});
    //     mask_matrix.attr("fill")(false);
    // });
}

#endif
