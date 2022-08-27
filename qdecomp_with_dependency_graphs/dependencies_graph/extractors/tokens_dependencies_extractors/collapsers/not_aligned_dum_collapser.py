from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, Span
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_spans import TokenData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class NotAlignedDumCollapser(BaseCollapser):
    def __init__(self, count=5):
        super().__init__()
        self.separator = '[DUM]'
        self.additional_tokens = [self.separator for _ in range(count)]

    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str) -> None:
        offset = len(spans_dependencies._tokens)
        spans_dependencies._tokens = list(spans_dependencies._tokens) + [
            TokenData(text=x, bio_tag=None)
            for x in self.additional_tokens
        ]
        empty_steps_ids = [i for i, data in spans_dependencies.steps() if not (data and data.spans_list)]
        assert len(empty_steps_ids) <= len(self.additional_tokens), \
            f'has no enough additional tokens  {self.separator}, has {len(self.additional_tokens)}, need {len(empty_steps_ids)}'

        dependencies_graph = spans_dependencies._dependencies_graph
        for i, step_id in enumerate(empty_steps_ids):
            span_data = dependencies_graph.nodes[step_id].get('data', SpansData([]))
            span_data.spans_list.append(Span(start=offset+i, end=offset+i))
            dependencies_graph.nodes[step_id]['data'] = span_data

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        sep_index = [i for i, x in enumerate(spans_dependencies.tokens()) if x.text == self.separator]
        for _, data in spans_dependencies.steps():
            for i in sep_index: data.remove_index(i)