#ifndef HTOOL_CLUSTER_CPP
#define HTOOL_CLUSTER_CPP

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "wrapper_mpi.hpp"
#include <htool/htool.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template <typename ClusterType>
void declare_Cluster(py::module &m, const std::string &className) {

    py::class_<ClusterType, std::shared_ptr<ClusterType>, VirtualCluster> py_class(m, className.c_str());
    py_class.def(py::init<int>());
    py_class.def(
        "build", [](ClusterType &self, int nb_pt, py::array_t<double, py::array::f_style | py::array::forcecast> x, int nb_sons, MPI_Comm_wrapper comm) {
            if (x.ndim() != 2 && x.shape()[0] != self.get_space_dim()) {
                throw std::runtime_error("Wrong dimension for x");
            }
            self.build(nb_pt, x.data(), nb_sons, comm);
        },
        "nb_pt"_a,
        "x"_a,
        "nb_sons"_a     = 2,
        py::arg("comm") = MPI_Comm_wrapper(MPI_COMM_WORLD));
    py_class.def(
        "build", [](ClusterType &self, int nb_pt, py::array_t<double, py::array::f_style | py::array::forcecast> x, py::array_t<int, py::array::f_style | py::array::forcecast> MasterOffset, int nb_sons, MPI_Comm_wrapper comm) {
            if (x.ndim() != 2 && x.shape()[0] != self.get_space_dim()) {
                throw std::runtime_error("Wrong dimension for x");
            }
            if (MasterOffset.ndim() != 2 && MasterOffset.shape()[0] != 2) {
                throw std::runtime_error("Wrong dimension for MasterOffset");
            }
            self.build(nb_pt, x.data(), MasterOffset.data(), nb_sons, comm);
        },
        "nb_pt"_a,
        "x"_a,
        "MasterOffset"_a,
        "nb_sons"_a     = 2,
        py::arg("comm") = MPI_Comm_wrapper(MPI_COMM_WORLD));
    py_class.def(
        "display", [](ClusterType &self, py::array_t<double, py::array::f_style | py::array::forcecast> x, int depth, MPI_Comm_wrapper comm) {
            int rankWorld;
            MPI_Comm_rank(comm, &rankWorld);

            if (rankWorld == 0) {

                VirtualCluster const *root = self.get_root();

                std::stack<VirtualCluster const *> s;
                s.push(root);

                int size      = root->get_size();
                int space_dim = root->get_space_dim();
                std::vector<double> output((space_dim + 1) * size);

                // Permuted geometric points
                for (int i = 0; i < size; ++i) {
                    for (int p = 0; p < space_dim; p++) {
                        output[i + size * p] = x.at(p, root->get_perm(i));
                    }
                }

                int counter = 0;
                while (!s.empty()) {
                    VirtualCluster const *curr = s.top();
                    s.pop();

                    if (depth == curr->get_depth()) {
                        std::fill_n(&(output[space_dim * size + curr->get_offset()]), curr->get_size(), counter);
                        counter += 1;
                    }

                    // Recursion
                    if (!curr->IsLeaf()) {

                        for (int p = 0; p < curr->get_nb_sons(); p++) {
                            s.push(&(curr->get_son(p)));
                        }
                    }
                }

                // Import
                py::object plt    = py::module::import("matplotlib.pyplot");
                py::object colors = py::module::import("matplotlib.colors");

                // Create Color Map
                py::object colormap = plt.attr("get_cmap")("Dark2");
                py::object norm     = colors.attr("Normalize")("vmin"_a = (*std::min_element(output.begin() + space_dim * size, output.end())), "vmax"_a = (*std::max_element(output.begin() + space_dim * size, output.end())));

                // Figure
                py::object fig = plt.attr("figure")();

                if (space_dim == 2) {
                    py::object ax = fig.attr("add_subplot")(111);
                    ax.attr("scatter")(std::vector<double>(output.begin(), output.begin() + size), std::vector<double>(output.begin() + size, output.begin() + 2 * size), "c"_a = colormap(norm(std::vector<double>(output.begin() + 2 * size, output.end()))), "marker"_a = 'o');

                } else if (space_dim == 3) {
                    py::object Axes3D = py::module::import("mpl_toolkits.mplot3d").attr("Axes3D");

                    py::object ax = fig.attr("add_subplot")(111, "projection"_a = "3d");
                    ax.attr("scatter")(std::vector<double>(output.begin(), output.begin() + size), std::vector<double>(output.begin() + size, output.begin() + 2 * size), std::vector<double>(output.begin() + 2 * size, output.begin() + 3 * size), "c"_a = colormap(norm(std::vector<double>(output.begin() + 3 * size, output.end()))), "marker"_a = 'o');
                }

                plt.attr("show")();
                return 0;
            }

            return 0;
        },
        "x"_a,
        "depth"_a,
        py::arg("comm") = MPI_Comm_wrapper(MPI_COMM_WORLD));
    py_class.def(
        "read_cluster", [](ClusterType &self, std::string file_permutation, std::string file_tree, MPI_Comm_wrapper comm) {
            self.read_cluster(file_permutation, file_tree, comm);
        },
        py::arg("file_permutation"),
        py::arg("file_tree"),
        py::arg("comm") = MPI_Comm_wrapper(MPI_COMM_WORLD));

    py_class.def("set_minclustersize", &ClusterType::set_minclustersize);
}

#endif