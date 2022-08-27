from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import DependencyType, SpansDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class ToDependencyTypeCollapser(BaseCollapser):
    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str= None) -> None:
        pass

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        for _, _, data in spans_dependencies.dependencies():
            data.dep_type = DependencyType(data.dep_type)
