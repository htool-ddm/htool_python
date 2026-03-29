#ifndef HTOOL_MATPLOTLIB_HMATRIX_CPP
#define HTOOL_MATPLOTLIB_HMATRIX_CPP

#include "../misc/logger.hpp"
#include <htool/hmatrix/hmatrix_output.hpp>
#include <pybind11/pybind11.h>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_matplotlib_hmatrix(py::module &m) {
    m.def("plot", [](py::object axes, const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, py::object L0) {
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
        py::object plt         = py::module::import("matplotlib.pyplot");
        py::object patches     = py::module::import("matplotlib.patches");
        py::object collections = py::module::import("matplotlib.collections");
        py::object colors      = py::module::import("matplotlib.colors");
        py::object numpy       = py::module::import("numpy");

        // Colormap
        py::object cmap     = plt.attr("get_cmap")("YlGn");
        py::object new_cmap = colors.attr("LinearSegmentedColormap").attr("from_list")("trunc(YlGn,0.4,1)", cmap(numpy.attr("linspace")(0.4, 1, 100)));
        py::object norm     = colors.attr("Normalize")("vmin"_a = 0, "vmax"_a = 10);

        // First Data
        int nr = hmatrix.get_target_cluster().get_size();
        int nc = hmatrix.get_source_cluster().get_size();

        // Storage for rectangles and colors
        py::list rects;
        py::list facecolors;

        for (int p = 0; p < nb_leaves; p++) {
            int i_row  = buf[5 * p];
            int nb_row = buf[5 * p + 1];
            int i_col  = buf[5 * p + 2];
            int nb_col = buf[5 * p + 3];
            int rank   = buf[5 * p + 4];

            // Color selection
            py::object facecolor;
            if (rank == -1) {
                facecolors.append(py::str("red")); // full blocks
            } else {
                facecolors.append(new_cmap(norm(rank)));
            }

            // Rectangle (no styling here!)
            py::object rect = patches.attr("Rectangle")(
                py::make_tuple(i_col, i_row),
                nb_col,
                nb_row);
            rects.append(rect);

            // Optional: annotate only large blocks
            if (rank >= 0 && nb_col > nc * 0.05 && nb_row > nr * 0.05) {
                axes.attr("text")(
                    i_col + nb_col / 2.0,
                    i_row + nb_row / 2.0,
                    rank,
                    "color"_a    = "white",
                    "ha"_a       = "center",
                    "va"_a       = "center",
                    "fontsize"_a = 8);
            }
        }

        // Create PatchCollection
        py::object collection = collections.attr("PatchCollection")(
            rects,
            "facecolor"_a = facecolors,
            "edgecolor"_a = "black", // huge speedup
            "linewidth"_a = 0.2);

        axes.attr("add_collection")(collection);

        // Axes formatting
        axes.attr("set_xlim")(0, nc);
        axes.attr("set_ylim")(nr, 0); // invert y-axis (matrix style)
        axes.attr("set_aspect")("equal");

        // Optional: remove ticks for speed
        axes.attr("set_xticks")(py::list());
        axes.attr("set_yticks")(py::list());

        if (!L0.is_none()) {
            py::list L0_rects;

            for (py::handle item : L0) {
                const HMatrix<CoefficientPrecision,CoordinatePrecision> &block =
                    item.cast<const HMatrix<CoefficientPrecision,CoordinatePrecision> &>();

                const int i_row =
                    block.get_target_cluster().get_offset()
                    - hmatrix.get_target_cluster().get_offset();

                const int nb_row =
                    block.get_target_cluster().get_size();

                const int i_col =
                    block.get_source_cluster().get_offset()
                    - hmatrix.get_source_cluster().get_offset();

                const int nb_col =
                    block.get_source_cluster().get_size();

                py::object rect =
                    patches.attr("Rectangle")(
                        py::make_tuple(i_col, i_row),
                        nb_col,
                        nb_row);

                L0_rects.append(rect);
            }

            py::object L0_collection =
                collections.attr("PatchCollection")(
                    L0_rects,
                    "facecolor"_a = "none",
                    "edgecolor"_a = "purple",
                    "linewidth"_a = 1.5);

            axes.attr("add_collection")(L0_collection);
        } }, py::arg("axes"), py::arg("hmatrix"), py::arg("L0") = py::none());
}

#endif
