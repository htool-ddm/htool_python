#ifndef HTOOL_HMATRIX_BUILDER_CPP
#define HTOOL_HMATRIX_BUILDER_CPP

#include "interfaces/virtual_dense_blocks_generator.hpp"
#include "interfaces/virtual_generator.hpp"
#include "interfaces/virtual_low_rank_generator.hpp"
#include <htool/hmatrix/tree_builder/tree_builder.hpp>

template <typename CoefficientPrecision, typename CoordinatePrecision>
void declare_hmatrix_builder(py::module &m, const std::string &className) {
    using Class = htool::HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>;
    py::class_<Class> py_class(m, className.c_str());

    // // Constructor
    // py_class.def(py::init([](underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta, char symmetry, char UPLO) {
    //                  return std::unique_ptr<Class>(new Class(epsilon, eta, symmetry, UPLO));
    //              }),
    //              py::arg("epsilon"),
    //              py::arg("eta"),
    //              py::arg("symmetry"),
    //              py::arg("UPLO"));

    py_class.def(py::init([](underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta, char symmetry, char UPLO, int reqrank, std::shared_ptr<VirtualLowRankGeneratorPython<CoefficientPrecision>> low_rank_strategy) {
                     std::cout << epsilon << "\n";
                     return std::unique_ptr<Class>(new Class(epsilon, eta, symmetry, UPLO, reqrank, low_rank_strategy));
                 }),
                 py::arg("epsilon"),
                 py::arg("eta"),
                 py::arg("symmetry"),
                 py::arg("UPLO"),
                 py::kw_only(),
                 py::arg("reqrank")           = -1,
                 py::arg("low_rank_strategy") = nullptr);

    // Build
    py_class.def("build", [](const Class &self, const VirtualGenerator<CoefficientPrecision> &generator, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, int target_partition_number, int partition_number_for_symmetry) { return self.build(generator, target_cluster, source_cluster, target_partition_number, partition_number_for_symmetry); }, py::arg("generator"), py::arg("target_cluster"), py::arg("source_cluster"), py::arg("target_partition_number") = -1, py::arg("partition_number_for_symmetry") = -1);

    // Setters
    py_class.def("set_minimal_source_depth", &Class::set_minimal_source_depth);
    py_class.def("set_minimal_target_depth", &Class::set_minimal_target_depth);
    py_class.def("set_low_rank_generator", [](Class &self, std::shared_ptr<VirtualLowRankGeneratorPython<CoefficientPrecision>> low_rank_generator) { self.set_low_rank_generator(low_rank_generator); });
    py_class.def("set_dense_blocks_generator", [](Class &self, std::shared_ptr<VirtualDenseBlocksGeneratorPython<CoefficientPrecision>> dense_blocks_generator) { self.set_dense_blocks_generator(dense_blocks_generator); });
}
#endif
