#ifndef HTOOL_OFF_DIAGONAL_APPROXIMATION_CPP
#define HTOOL_OFF_DIAGONAL_APPROXIMATION_CPP

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <htool/htool.hpp>
#include <htool/types/off_diagonal_approximation_with_hmatrix.hpp>

template <typename T>
class VirtualOffDiagonalApproximationCpp : public VirtualOffDiagonalApproximation<T> {
    int nc, nr;

  public:
    using VirtualOffDiagonalApproximation<T>::VirtualOffDiagonalApproximation;

    VirtualOffDiagonalApproximationCpp(const VirtualHMatrix<T> &HA) {
        HA.get_off_diagonal_size(nr, nc);
    }

    void mvprod_global_to_local(const T *const in, T *const out, const int &mu) override {

        py::array_t<T, py::array::c_style> in_py(std::array<long int, 2>{nc, mu}, in, py::capsule(in));
        py::array_t<T, py::array::c_style> out_py(std::array<long int, 2>{nr, mu}, out, py::capsule(out));

        mat_mat_prod_global_to_local(in_py, out_py);
        // for (int i = 0; i < nr; i++) {
        //     for (int j = 0; j < mu; j++) {
        //         std::cout << out[i * mu + j] << " " << out_py.at(i, j) << " ";
        //     }
        //     std::cout << "\n";
        // }
    }

    void mvprod_subrhs_to_local(const T *const in, T *const out, const int &mu, const int &offset, const int &size) override {
        py::array_t<T, py::array::c_style> in_py(std::array<long int, 2>{size, mu}, in, py::capsule(in));
        py::array_t<T, py::array::c_style> out_py(std::array<long int, 2>{nr, mu}, out, py::capsule(out));

        mat_mat_prod_sub_rhs_to_local(in_py, out_py, mu, offset, size);
    }

    // lcov does not see it because of trampoline I assume
    virtual void mat_mat_prod_global_to_local(const py::array_t<T, py::array::c_style> &in, py::array_t<T, py::array::c_style> &out) const = 0; // LCOV_EXCL_LINE

    virtual void mat_mat_prod_sub_rhs_to_local(const py::array_t<T, py::array::c_style> &in, py::array_t<T, py::array::c_style> &out, int mu, int offset, int size) const {
        std::vector<T> in_global(nc * mu, 0);
        std::copy_n(in.data(), size * mu, in_global.data());
        py::array_t<T, py::array::c_style> in_global_pyarray({nc, mu}, in_global.data());

        this->mat_mat_prod_global_to_local(in_global_pyarray, out);
    }
};

template <typename T>
class PyVirtualOffDiagonalApproximation : public VirtualOffDiagonalApproximationCpp<T> {
  public:
    using VirtualOffDiagonalApproximationCpp<T>::VirtualOffDiagonalApproximationCpp;
    // PyVirtualGenerator(int nr0, int nc0): IMatrix<T>(nr0,nc0){}

    /* Trampoline (need one for each virtual function) */
    virtual void mat_mat_prod_global_to_local(const py::array_t<T, py::array::c_style> &in, py::array_t<T, py::array::c_style> &out) const override {
        PYBIND11_OVERRIDE_PURE(
            void,                                  /* Return type */
            VirtualOffDiagonalApproximationCpp<T>, /* Parent class */
            mat_mat_prod_global_to_local,          /* Name of function in C++ (must match Python name) */
            in,
            out /* Argument(s) */
        );
    }

    /* Trampoline (need one for each virtual function) */
    virtual void mat_mat_prod_sub_rhs_to_local(const py::array_t<T, py::array::c_style> &in, py::array_t<T, py::array::c_style> &out, int mu, int offset, int size) const override {
        PYBIND11_OVERRIDE(
            void,                                  /* Return type */
            VirtualOffDiagonalApproximationCpp<T>, /* Parent class */
            mat_mat_prod_sub_rhs_to_local,         /* Name of function in C++ (must match Python name) */
            in,
            out,
            mu,
            offset,
            size /* Argument(s) */
        );
    }
};

template <typename T>
void declare_VirtualOffDiagonalApproximation(py::module &m, const std::string &BaseClassName, const std::string &VirtualClassName, const std::string &ClassName) {
    // Not to be used, but we need to declare it to use it as an argument
    using VirtualBaseClass = VirtualOffDiagonalApproximation<T>;
    py::class_<VirtualBaseClass, std::shared_ptr<VirtualBaseClass>> py_base_class(m, BaseClassName.c_str());

    // Virtual class that the user can use
    using VirtualClass = VirtualOffDiagonalApproximationCpp<T>;
    py::class_<VirtualClass, std::shared_ptr<VirtualClass>, PyVirtualOffDiagonalApproximation<T>, VirtualBaseClass> py_virtual_class(m, VirtualClassName.c_str());
    py_virtual_class.def(py::init<const VirtualHMatrix<T> &>());
    py_virtual_class.def("mat_mat_prod_global_to_local", &VirtualClass::mat_mat_prod_global_to_local);

    // Possible off diagonal approximation defined in htool
    using Class = OffDiagonalApproximationWithHMatrix<T>;
    py::class_<Class, std::shared_ptr<Class>, VirtualBaseClass> py_class(m, ClassName.c_str());
    py_class.def(py::init<VirtualHMatrix<T> *, std::shared_ptr<VirtualCluster>, std::shared_ptr<VirtualCluster>>());
    py_class.def("build", [](Class &self, VirtualGeneratorCpp<T> &mat, const py::array_t<double, py::array::f_style> &x, const py::array_t<double, py::array::f_style> &y) {
        self.build(mat, x.data(), y.data());
    });

    // Plot pattern
    py_class.def(
        "display", [](const Class &self, bool show = true) {
            std::vector<int> buf = self.get_output();
            int nbg              = buf.size() / 5;

            // Import
            py::object plt     = py::module::import("matplotlib.pyplot");
            py::object patches = py::module::import("matplotlib.patches");
            py::object colors  = py::module::import("matplotlib.colors");
            py::object numpy   = py::module::import("numpy");

            // First Data
            int nr = self.nb_rows();
            int nc = self.nb_cols();
            py::array_t<int> matrix({nr, nc});
            matrix.attr("fill")(0);
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
            for (int p = 0; p < nbg; p++) {
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
