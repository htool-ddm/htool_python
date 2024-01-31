#ifndef HTOOL_MATPLOTLIB_HMATRIX_CPP
#define HTOOL_MATPLOTLIB_HMATRIX_CPP

#include "../misc/logger.hpp"
#include <htool/hmatrix/hmatrix_output.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_matplotlib_hmatrix(py::module &m) {
    m.def("plot", [](py::object axes, const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
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
        matrix.attr("fill")(0);
        mask_matrix.attr("fill")(false);

        // Figure
        // py::tuple sublots_output = plt.attr("subplots")(1, 1);
        // py::object fig           = sublots_output[0];
        // py::object axes          = sublots_output[1];
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

        axes.attr("imshow")(masked_matrix, "cmap"_a = new_cmap, "vmin"_a = 0, "vmax"_a = 10);
        // plt.attr("draw")();
    });
}

#endif
