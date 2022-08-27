from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, Span, SpanDependencyData
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_spans import TokenData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class LastStepCollapser(BaseCollapser):
    def __init__(self, create_separate_span: bool):
        """
        Add a spacial token [ROOT] that is aligned to the last span
        :param create_separate_span: boolean, if True creates a new span that refers with a 'root' dependency to the
        last span. Otherwise, join the special token to the (current) last span.
        """
        super().__init__()
        self.separator = '[ROOT]'
        self.dependency = 'root' if create_separate_span else None

    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str=None) -> None:
        sep_index = len(spans_dependencies._tokens)
        spans_dependencies._tokens = list(spans_dependencies._tokens) + [TokenData(text=self.separator, bio_tag=None)]

        dependencies_graph = spans_dependencies._dependencies_graph
        last_step_id = max(i for i,_ in spans_dependencies.steps())

        if self.dependency:
            # create a separate span that points with a new dependency to the last one
            dependencies_graph.add_node(last_step_id+1, data=SpansData([Span(start=sep_index, end=sep_index)]))
            dependencies_graph.add_edge(last_step_id+1, last_step_id, data=SpanDependencyData(dep_type=self.dependency))
        else:
            # align separator to the last span
            span_data = dependencies_graph.nodes[last_step_id].get('data', SpansData([]))
            span_data.spans_list.append(Span(start=sep_index, end=sep_index))
            dependencies_graph.nodes[last_step_id]['data'] = span_data

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        dependencies_graph = spans_dependencies._dependencies_graph

        if self.dependency:
            # remove root node
            root_edge_src = [u for (u,v,k,data) in dependencies_graph.edges(data=True, keys=True)
                             if data['data'].dep_type == self.dependency]
            dependencies_graph.remove_nodes_from(root_edge_src)
        else:
            sep_index = [i for i, x in enumerate(spans_dependencies.tokens()) if x.text == self.separator]
            assert len(sep_index) == 1, f"should by exactly one {self.separator} token. got {len(sep_index)}"
            for _, data in spans_dependencies.steps():
                data.spans_list = [x for x in data.spans_list if sep_index[0] not in [x.start, x.end]]