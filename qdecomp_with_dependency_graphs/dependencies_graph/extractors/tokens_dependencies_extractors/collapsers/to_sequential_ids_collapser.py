import networkx as nx
from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class ToSequentialIdsCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        pass

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        dependencies_graph = spans_dependencies._dependencies_graph
        # fix steps ids
        relabel_map = {n_id: i for (n_id, i) in
                       zip(sorted(dependencies_graph.nodes()), range(1, dependencies_graph.number_of_nodes() + 1))
                       if n_id != i}
        if relabel_map:
            nx.relabel_nodes(dependencies_graph, relabel_map, copy=False)
