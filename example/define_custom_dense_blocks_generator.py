import Htool


class CustomDenseBlocksGenerator(Htool.VirtualDenseBlocksGenerator):
    def __init__(
        self, generator, target_cluster: Htool.Cluster, source_cluster: Htool.Cluster
    ):
        super().__init__(target_cluster, source_cluster)
        self.generator = generator

    def build_dense_blocks(self, rows_offsets, cols_offsets, blocks):
        nb_blocks = len(blocks)
        for i in range(nb_blocks):
            self.generator.build_submatrix(rows_offsets[i], cols_offsets[i], blocks[i])
