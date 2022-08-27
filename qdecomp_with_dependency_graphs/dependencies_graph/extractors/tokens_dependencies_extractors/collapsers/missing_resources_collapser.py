from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, Span
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_spans import TokenData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser
from qdecomp_with_dependency_graphs.scripts.data_processing.break_app_store_generation import get_additional_resources


class MissingResourcesCollapser(BaseCollapser):
    def __init__(self):
        super().__init__()
        self.separator = '[RSC]'
        self.additional_tokens = [self.separator] + get_additional_resources()

    @overrides
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str) -> None:
        offset = len(spans_dependencies._tokens) + 1
        spans_dependencies._tokens = list(spans_dependencies._tokens) + [
            TokenData(text=x, bio_tag=None)
            for x in self.additional_tokens
        ]
        empty_steps_ids = [i for i, data in spans_dependencies.steps() if not (data and data.spans_list)]

        dependencies_graph = spans_dependencies._dependencies_graph
        steps_tokens = [x.split(' ') for x in decomposition.split(';')]
        for i in empty_steps_ids:
            for j, add_token in enumerate(self.additional_tokens[1:]):
                if add_token in steps_tokens[i-1]:
                    span_data = dependencies_graph.nodes[i].get('data', SpansData([]))
                    span_data.spans_list.append(Span(start=offset+j, end=offset+j))
                    dependencies_graph.nodes[i]['data'] = span_data

    @overrides
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        pass