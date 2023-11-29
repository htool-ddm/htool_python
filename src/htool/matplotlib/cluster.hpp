#ifndef HTOOL_MATPLOTLIB_CLUSTER_CPP
#define HTOOL_MATPLOTLIB_CLUSTER_CPP

#include "../misc/logger.hpp"
#include <pybind11/pybind11.h>

template <typename CoordinatePrecision>
void declare_matplotlib_cluster(py::module &m) {
    m.def("plot", [](py::object ax, Cluster<CoordinatePrecision> &cluster, py::array_t<double, py::array::f_style | py::array::forcecast> coordinates, int depth) {
        int spatial_dimension   = coordinates.shape()[0];
        int number_of_points    = coordinates.shape()[1];
        int root_cluster_size   = cluster.get_size();
        int root_cluster_offset = cluster.get_offset();
        const auto &permutation = cluster.get_permutation();

        // Runtime checks
        if (spatial_dimension == 3 && ax.attr("name").cast<std::string>() != "3d") {
            htool::Logger::get_instance()
                .log(LogLevel::WARNING, "Axes object is not 3d while coordinates are 3d.");
        }
        if (spatial_dimension == 2 && ax.attr("name").cast<std::string>() != "rectilinear") {
            htool::Logger::get_instance()
                .log(LogLevel::WARNING, "Axes object is not rectilinear while coordinates are 2d.");
        }

        std::vector<double> output((spatial_dimension)*root_cluster_size);
        std::vector<int> partition_numbers(root_cluster_size);
        double counter = 0;

        // Permuted geometric points
        for (int i = root_cluster_offset; i < root_cluster_offset + root_cluster_size; ++i) {
            for (int p = 0; p < spatial_dimension; p++) {
                output[(i - root_cluster_offset) + root_cluster_size * p] = coordinates.at(p, permutation[i]);
            }
        }

        preorder_tree_traversal(
            cluster,
            [&partition_numbers, &counter, &depth, &root_cluster_offset, &root_cluster_size](const Cluster<CoordinatePrecision> &current_cluster) {
                if (current_cluster.get_depth() == depth) {
                    std::fill_n(partition_numbers.begin() + current_cluster.get_offset() - root_cluster_offset, current_cluster.get_size(), counter);
                    counter += 1;
                }
            });

        // Import
        py::object plt    = py::module::import("matplotlib.pyplot");
        py::object colors = py::module::import("matplotlib.colors");

        // Create Color Map
        py::object colormap = plt.attr("get_cmap")("Dark2");
        if (counter == 9) {
            colormap = plt.attr("get_cmap")("Set1");
        }
        if (counter == 10) {
            colormap = plt.attr("get_cmap")("tab10");
        }
        if (counter > 10 && counter <= 20) {
            colormap = plt.attr("get_cmap")("tab20");
        }
        if (counter > 20) {
            htool::Logger::get_instance()
                .log(LogLevel::WARNING, "Colormap does not support more than 20 colors.");
        }

        py::object norm = colors.attr("Normalize")("vmin"_a = (*std::min_element(partition_numbers.begin(), partition_numbers.end())), "vmax"_a = (*std::max_element(partition_numbers.begin(), partition_numbers.end())));

        // Figure
        if (spatial_dimension == 2) {
            ax.attr("scatter")(std::vector<double>(output.begin(), output.begin() + root_cluster_size), std::vector<double>(output.begin() + root_cluster_size, output.end()), "c"_a = colormap(norm(std::vector<double>(partition_numbers.begin(), partition_numbers.end()))), "marker"_a = 'o');

        } else if (spatial_dimension == 3) {
            ax.attr("scatter")(std::vector<double>(output.begin(), output.begin() + root_cluster_size), std::vector<double>(output.begin() + root_cluster_size, output.begin() + 2 * root_cluster_size), std::vector<double>(output.begin() + 2 * root_cluster_size, output.end()), "c"_a = colormap(norm(std::vector<double>(partition_numbers.begin(), partition_numbers.end()))), "marker"_a = 'o');
        }
    });
}

#endif
