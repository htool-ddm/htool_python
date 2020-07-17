#ifndef HTOOL_HMATRIX_CPP
#define HTOOL_HMATRIX_CPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h> 

#include <htool/htool.hpp>
#include "wrapper_mpi.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace htool;

template<typename T, template<typename,typename> class LowRankMatrix, class ClusterImpl>
void declare_HMatrix(py::module &m, const std::string& className) {
    using Class = HMatrix<T, LowRankMatrix, ClusterImpl>;
    py::class_<Class> py_class(m, className.c_str()) ;

    // Symmetric build
    py_class.def(py::init<IMatrix<T>&,const std::vector<R3>&,bool, const int&,  MPI_Comm_wrapper>(),py::arg("mat"), py::arg("xt"), py::arg("Symmetry") = false, py::arg("reqrank") = -1, py::arg("comm")=MPI_Comm_wrapper(MPI_COMM_WORLD));

    // Non symmetric build
    py_class.def(py::init<IMatrix<T>&, const std::vector<R3>& , const std::vector<R3>&, const int&, MPI_Comm_wrapper>(),py::arg("mat"), py::arg("xt"),py::arg("xs"), py::arg("reqrank") = -1, py::arg("comm")=MPI_Comm_wrapper(MPI_COMM_WORLD));

    // Getters
    py_class.def_property_readonly("shape",[](const Class &self) {
        return std::array<int,2>{self.nb_rows(),self.nb_cols()};
    });


    // Linear algebra
    py_class.def("__mul__",[](const Class &self, std::vector<T> b) {
        return self * b;
    });
    py_class.def("matvec",[](const Class &self, std::vector<T> b) {
        return self * b;
    });
    py_class.def("__matmul__",[](const Class &self, py::array_t<T,  py::array::f_style | py::array::forcecast> B) {
        int mu;

        if (B.ndim()==1){
            mu = 1;
        } else if (B.ndim()==2){
            mu = B.shape()[1];
        }
        else if (B.ndim()!=1){
            throw std::runtime_error("Wrong dimension for HMatrix-matrix product");
        }
        if (B.shape()[0]!=self.nb_cols()){
            throw std::runtime_error("Wrong size for HMatrix-matrix product");
        }


        std::vector<T> result(self.nb_rows()*mu,0);


        self.mvprod_global(B.data(),result.data(),mu);

        if (B.ndim()==1){
            std::array<long int, 1> shape {self.nb_rows()};
            return py::array_t<T,py::array::f_style> (shape,result.data());
        }
        else {
            std::array<long int, 2> shape {self.nb_rows(), mu};
            return py::array_t<T,py::array::f_style> (shape,result.data());  
        }
    });




    // Print information
    py_class.def("print_infos",&Class::print_infos);
    py_class.def("__str__",[](const Class &self){
        return "HMatrix: (shape: "+ htool::NbrToStr(self.nb_cols())+"x"+htool::NbrToStr(self.nb_rows())+", nb_low_rank_blocks: "+ htool::NbrToStr(self.get_nlrmat())+", nb_dense_blocks: "+ htool::NbrToStr(self.get_ndmat())+")";
    });

    // Plot pattern
    py_class.def("display",[](const Class &self){
        
        const std::vector<LowRankMatrix<T,ClusterImpl>*>& lrmats = self.get_MyFarFieldMats();
	    const std::vector<SubMatrix<T>*>& dmats = self.get_MyNearFieldMats();

        

	    int nb = dmats.size() + lrmats.size();
        int sizeworld = self.get_sizeworld();
        int rankworld = self.get_rankworld();

        int nbworld[sizeworld];
        MPI_Allgather(&nb, 1, MPI_INT, nbworld, 1, MPI_INT, self.get_comm());
        int nbg = 0;
        for (int i=0; i<sizeworld; i++) {
            nbg += nbworld[i];
        }

        std::vector<int> buf(5*nbg,0);

        for (int i=0;i<dmats.size();i++) {
            const SubMatrix<T>& l = *(dmats[i]);
            buf[5*i] = l.get_offset_i();
            buf[5*i+1] = l.nb_rows();
            buf[5*i+2] = l.get_offset_j();
            buf[5*i+3] = l.nb_cols();
            buf[5*i+4] = -1;
        }

        for (int i=0;i<lrmats.size();i++) {
            const LowRankMatrix<T,ClusterImpl>& l = *(lrmats[i]);
            buf[5*(dmats.size()+i)] = l.get_offset_i();
            buf[5*(dmats.size()+i)+1] = l.nb_rows();
            buf[5*(dmats.size()+i)+2] = l.get_offset_j();
            buf[5*(dmats.size()+i)+3] = l.nb_cols();
            buf[5*(dmats.size()+i)+4] = l.rank_of();
        }

        int displs[sizeworld];
        int recvcounts[sizeworld];
        displs[0] = 0;

        for (int i=0; i<sizeworld; i++) {
            recvcounts[i] = 5*nbworld[i];
            if (i > 0)	displs[i] = displs[i-1] + recvcounts[i-1];
        }
        MPI_Gatherv(rankworld==0 ? MPI_IN_PLACE:buf.data(), recvcounts[rankworld], MPI_INT, buf.data(), recvcounts, displs, MPI_INT, 0, self.get_comm());
        

        if (rankworld==0){
            // Import
            py::object plt = py::module::import("matplotlib.pyplot");
            py::object patches = py::module::import("matplotlib.patches");
            py::object colors = py::module::import("matplotlib.colors");
            py::object numpy = py::module::import("numpy");

            // First Data
            int nr = self.nb_rows();
            int nc = self.nb_cols();
            py::array_t<int> matrix({nr,nc});
            py::array_t<bool> mask_matrix({nr,nc});
            mask_matrix.attr("fill")(false);

            // Figure
            py::tuple sublots_output = plt.attr("subplots")(1,1);
            py::object fig  = sublots_output[0];
            py::object axes = sublots_output[1];
            // axes.attr()

            // Issue: there a shift of one pixel along the y-axis...
            // int shift = axes.transData.transform([(0,0), (1,1)])
            // shift = shift[1,1] - shift[0,1]  # 1 unit in display coords
            int shift = 0;

            int max_rank=0;
            for (int p=0;p<nbg;p++){
                int i_row= buf[5*p];
                int nb_row = buf[5*p+1];
                int i_col = buf[5*p+2];
                int nb_col = buf[5*p+3];
                int rank = buf[5*p+4]; 

                if (rank>max_rank){
                    max_rank=rank;
                }
                for (int i=0;i<nb_row;i++){
                    for (int j=0;j<nb_col;j++){
                        matrix.mutable_at(i_row+i,i_col+j)=rank;
                        if (rank==-1){
                            mask_matrix.mutable_at(i_row+i,i_col+j)=true;
                        }
                    }
                }

                py::object rect = patches.attr("Rectangle")(py::make_tuple(i_col-0.5,i_row-0.5+shift),nb_col,nb_row,"linewidth"_a=0.75,"edgecolor"_a='k',"facecolor"_a="none");
                axes.attr("add_patch")(rect);

                if (rank>=0 && nb_col/double(nc)>0.05 && nb_row/double(nc)>0.05){
                    axes.attr("annotate")(rank,py::make_tuple(i_col+nb_col/2.,i_row+nb_row/2.),"color"_a="white","size"_a=10, "va"_a="center", "ha"_a="center");
                }
            }

            // Colormap
            py::object cmap = plt.attr("get_cmap")("YlGn");
            py::object new_cmap = colors.attr("LinearSegmentedColormap").attr("from_list")("trunc(YlGn,0.4,1)", cmap(numpy.attr("linspace")(0.4, 1, 100)));



            // Plot
            py::object masked_matrix =numpy.attr("ma").attr("array")(matrix,"mask"_a=mask_matrix);
            new_cmap.attr("set_bad")("color"_a="red");

            plt.attr("imshow")(masked_matrix,"cmap"_a=new_cmap,"vmin"_a=0, "vmax"_a=10);
            plt.attr("show")();
        }
    });

    // Plot clustering

    py_class.def("display_cluster",[](const Class &self,std::vector<R3> points_target, int depth, std::string type ){
        int sizeworld = self.get_sizeworld();
        int rankworld = self.get_rankworld();

        if (rankworld==0){
            
            Cluster<GeometricClustering> const * root ;
            if (type=="target"){
                root = &(self.get_cluster_tree_t());
            }
            else if (type=="source"){
                root = &(self.get_cluster_tree_s());
            }
            else {
                std::cout << "Choose between target and source"<< std::endl;
                return 0;
            }
            
            std::stack< Cluster<GeometricClustering> const *> s;
            s.push(root);

            int size = root->get_size();
            std::vector<double> output(4*size);

            // Permuted geometric points
            for(int i = 0; i<size; ++i) {
                output[i  ]     = points_target[root->get_perm(i)][0];
                output[i+size]  = points_target[root->get_perm(i)][1];
                output[i+size*2]= points_target[root->get_perm(i)][2];
            }

            int counter = 0;
            while(!s.empty()){
                Cluster<GeometricClustering> const * curr = s.top();
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


    },py::arg("points_target"),py::arg("depth"),py::arg("type")="target");
}



#endif