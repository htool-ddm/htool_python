#ifndef HTOOL_CLUSTER_CPP
#define HTOOL_CLUSTER_CPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h> 

#include <htool/htool.hpp>
#include "wrapper_mpi.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template<typename ClusterType>
void declare_Cluster(py::module &m, const std::string& className) {
    py::class_<Cluster<ClusterType>, std::shared_ptr<Cluster<ClusterType>>>(m, className.c_str())
        .def(py::init<>())
        .def("build", [](Cluster<ClusterType>& self,const std::vector<R3>&x, const std::vector<double>&r, const std::vector<int>& tab, const std::vector<double>& g, int nb_sons, MPI_Comm_wrapper comm){
            self.build(x,r,tab,g,nb_sons,comm);
        },"x"_a, "r"_a, "tab"_a, "g"_a, "nb_sons"_a=2, py::arg("comm")=MPI_Comm_wrapper(MPI_COMM_WORLD))
        .def("display", [](Cluster<ClusterType>& self,const std::vector<R3>&x, int depth, MPI_Comm_wrapper comm){

            int rankWorld;
            MPI_Comm_rank(comm, &rankWorld);

            if (rankWorld==0){
                
                Cluster<ClusterType> const * root =&(self.get_root());
                
                std::stack< Cluster<ClusterType> const *> s;
                s.push(root);

                int size = root->get_size();
                std::vector<double> output(4*size);

                // Permuted geometric points
                for(int i = 0; i<size; ++i) {
                    output[i  ]     = x[root->get_perm(i)][0];
                    output[i+size]  = x[root->get_perm(i)][1];
                    output[i+size*2]= x[root->get_perm(i)][2];
                }

                int counter = 0;
                while(!s.empty()){
                    Cluster<ClusterType> const * curr = s.top();
                    s.pop();

                    if (depth == curr->get_depth()){
                        std::fill_n(&(output[3*size+curr->get_offset()]),curr->get_size(),counter);
                        counter+=1;
                    }

                    // Recursion
                    if (!curr->IsLeaf()){
                        
                        for (int p=0;p<curr->get_nb_sons();p++){
                            s.push(&(curr->get_son(p)));
                        }
                    }
                }

                // Import
                py::object plt = py::module::import("matplotlib.pyplot");
                py::object colors = py::module::import("matplotlib.colors");
                py::object Axes3D = py::module::import("mpl_toolkits.mplot3d").attr("Axes3D");

                // Create Color Map
                py::object colormap = plt.attr("get_cmap")("Dark2");
                py::object norm    = colors.attr("Normalize")("vmin"_a=(*std::min_element(output.begin()+3*size,output.end())), "vmax"_a=(*std::max_element(output.begin()+3*size,output.end())));

                // Figure
                py::object fig = plt.attr("figure")();
                py::object ax = fig.attr("add_subplot")(111, "projection"_a="3d");

                ax.attr("scatter")(std::vector<double>(output.begin(),output.begin()+size), std::vector<double>(output.begin()+size,output.begin()+2*size), std::vector<double>(output.begin()+2*size,output.begin()+3*size),"c"_a=colormap(norm(std::vector<double>(output.begin()+3*size,output.end()))), "marker"_a='o');

                plt.attr("show")();
                return 0;

            }

            return 0;


        },"x"_a, "depth"_a, py::arg("comm")=MPI_Comm_wrapper(MPI_COMM_WORLD))
        .def("read_cluster",[](Cluster<ClusterType>& self,std::string file_permutation,std::string file_tree, MPI_Comm_wrapper comm){
            self.read_cluster(file_permutation,file_tree,comm);

        },py::arg("file_permutation"),py::arg("file_tree"),py::arg("comm")=MPI_Comm_wrapper(MPI_COMM_WORLD));
}

#endif