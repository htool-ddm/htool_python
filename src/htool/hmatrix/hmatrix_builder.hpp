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

    // Constructor
    py_class.def(py::init<const htool::Cluster<CoordinatePrecision> &, const htool::Cluster<CoordinatePrecision> &, htool::underlying_type<CoefficientPrecision>, CoordinatePrecision, char, char, int, int, int>());

    // Build
    // py_class.def("build", [](const Class &self, const VirtualGenerator<CoefficientPrecision> &generator) { return self.build(generator); });
    py_class.def("build", [](const Class &self, const VirtualGenerator<CoefficientPrecision> &generator) { return self.build(InternalGeneratorWithPermutation<CoefficientPrecision>(generator, self.get_target_cluster().get_permutation().data(), self.get_source_cluster().get_permutation().data())); });

    // Setters
    py_class.def("set_minimal_source_depth", &Class::set_minimal_source_depth);
    py_class.def("set_minimal_target_depth", &Class::set_minimal_target_depth);
    py_class.def("set_low_rank_generator", [](Class &self, const std::shared_ptr<VirtualLowRankGeneratorPython<CoefficientPrecision, CoordinatePrecision>> &low_rank_generator) { self.set_low_rank_generator(low_rank_generator); });
    py_class.def("set_dense_blocks_generator", [](Class &self, const std::shared_ptr<VirtualDenseBlocksGeneratorPython<CoefficientPrecision>> &dense_blocks_generator) { self.set_dense_blocks_generator(dense_blocks_generator); });
}
#endif
