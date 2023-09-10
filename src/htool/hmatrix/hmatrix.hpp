#ifndef HTOOL_HMATRIX_CPP
#define HTOOL_HMATRIX_CPP

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_HMatrix(py::module &m, const std::string &className) {

    using Class = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());

    // py_class.def("build", [](Class &self, VirtualGeneratorCpp<T> &mat, const py::array_t<double, py::array::f_style> &x) {
    //     self.build(mat, x.data());
    // });

    // py_class.def("build_dense_blocks", [](Class &self, VirtualDenseBlocksGeneratorCpp<T> &dense_block_generator) {
    //     self.build_dense_blocks(dense_block_generator);
    // });

    // // Setters
    // py_class.def("set_maxblocksize", &Class::set_maxblocksize);
    // py_class.def("set_minsourcedepth", &Class::set_minsourcedepth);
    // py_class.def("set_mintargetdepth", &Class::set_mintargetdepth);
    // py_class.def("set_delay_dense_computation", &Class::set_delay_dense_computation);
    // py_class.def("set_compression", [](Class &self, std::shared_ptr<VirtualLowRankGeneratorCpp<T>> mat) {
    //     self.set_compression(mat);
    // });

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

    // // Print information
    // py_class.def("print_infos", &Class::print_infos);
    // py_class.def("get_infos", overload_cast_<const std::string &>()(&Class::get_infos, py::const_));
    // py_class.def("__str__", [](const Class &self) {
    //     return "HMatrix: (shape: " + htool::NbrToStr(self.nb_cols()) + "x" + htool::NbrToStr(self.nb_rows()) + ", nb_low_rank_blocks: " + htool::NbrToStr(self.get_nlrmat()) + ", nb_dense_blocks: " + htool::NbrToStr(self.get_ndmat()) + ")";
    // });

    py_class.def(
        "get_block_diagonal_hmatrix", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
            return &*hmatrix.get_diagonal_hmatrix();
        },
        py::return_value_policy::reference_internal);
    py_class.def("get_tree_parameters", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
        std::stringstream ss;
        htool::print_tree_parameters(hmatrix, ss);

        return ss.str();
    });
    py_class.def("get_information", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
        std::stringstream ss;
        htool::print_hmatrix_information(hmatrix, ss);

        return ss.str();
    });

    // Plot pattern
    py_class.def(
        "display", [](const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, bool show = true) {
            std::vector<int> buf;
            int nb_leaves = 0;
            htool::preorder_tree_traversal(
                hmatrix,
                [&buf, &nb_leaves, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatePrecision> &current_hmatrix) {
                    if (current_hmatrix.is_leaf()) {
                        nb_leaves += 1;
                        buf.push_back(current_hmatrix.get_target_cluster().get_offset() - hmatrix.get_target_cluster().get_offset());
                        buf.push_back(current_hmatrix.get_target_cluster().get_size());
                        buf.push_back(current_hmatrix.get_source_cluster().get_offset() - hmatrix.get_source_cluster().get_offset());
                        buf.push_back(current_hmatrix.get_source_cluster().get_size());
                        buf.push_back(current_hmatrix.get_rank());
                    }
                });

            // Import
            py::object plt     = py::module::import("matplotlib.pyplot");
            py::object patches = py::module::import("matplotlib.patches");
            py::object colors  = py::module::import("matplotlib.colors");
            py::object numpy   = py::module::import("numpy");

            // First Data
            int nr = hmatrix.get_target_cluster().get_size();
            int nc = hmatrix.get_source_cluster().get_size();
            py::array_t<int> matrix({nr, nc});
            py::array_t<bool> mask_matrix({nr, nc});
            mask_matrix.attr("fill")(false);

            // Figure
            py::tuple sublots_output = plt.attr("subplots")(1, 1);
            py::object fig           = sublots_output[0];
            py::object axes          = sublots_output[1];
            // axes.attr()

            // Issue: there a shift of one pixel along the y-axis...
            // int shift = axes.transData.transform([(0,0), (1,1)])
            // shift = shift[1,1] - shift[0,1]  # 1 unit in display coords
            int shift = 0;

            int max_rank = 0;
            for (int p = 0; p < nb_leaves; p++) {
                int i_row  = buf[5 * p];
                int nb_row = buf[5 * p + 1];
                int i_col  = buf[5 * p + 2];
                int nb_col = buf[5 * p + 3];
                int rank   = buf[5 * p + 4];

                if (rank > max_rank) {
                    max_rank = rank;
                }
                for (int i = 0; i < nb_row; i++) {
                    for (int j = 0; j < nb_col; j++) {
                        matrix.mutable_at(i_row + i, i_col + j) = rank;
                        if (rank == -1) {
                            mask_matrix.mutable_at(i_row + i, i_col + j) = true;
                        }
                    }
                }

                py::object rect = patches.attr("Rectangle")(py::make_tuple(i_col - 0.5, i_row - 0.5 + shift), nb_col, nb_row, "linewidth"_a = 0.75, "edgecolor"_a = 'k', "facecolor"_a = "none");
                axes.attr("add_patch")(rect);

                if (rank >= 0 && nb_col / double(nc) > 0.05 && nb_row / double(nc) > 0.05) {
                    axes.attr("annotate")(rank, py::make_tuple(i_col + nb_col / 2., i_row + nb_row / 2.), "color"_a = "white", "size"_a = 10, "va"_a = "center", "ha"_a = "center");
                }
            }

            // Colormap
            py::object cmap     = plt.attr("get_cmap")("YlGn");
            py::object new_cmap = colors.attr("LinearSegmentedColormap").attr("from_list")("trunc(YlGn,0.4,1)", cmap(numpy.attr("linspace")(0.4, 1, 100)));

            // Plot
            py::object masked_matrix = numpy.attr("ma").attr("array")(matrix, "mask"_a = mask_matrix);
            new_cmap.attr("set_bad")("color"_a = "red");

            plt.attr("imshow")(masked_matrix, "cmap"_a = new_cmap, "vmin"_a = 0, "vmax"_a = 10);
            plt.attr("draw")();
            if (show) {
                plt.attr("show")(); // LCOV_EXCL_LINE
            }
        },
        py::arg("show") = true);
}

#endif
