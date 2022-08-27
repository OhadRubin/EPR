from typing import List, Iterable
from overrides import overrides

import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokenDependencyType, \
    TokenData, TokenDependencyData, TokensDependencies, SpansDependencies

from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors import BaseTokensDependenciesExtractor
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.spans_dependencies_extractors import BaseSpansDependenciesExtractor
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors.collapsers.base_collapser import BaseCollapser


class TokensDependenciesExtractor(BaseTokensDependenciesExtractor):
    def __init__(self, spans_dependencies_extractor: BaseSpansDependenciesExtractor, collapsers: List[BaseCollapser] = None):
        self.spans_dependencies_extractor = spans_dependencies_extractor
        self.collapsers = collapsers

    @overrides
    def extract(self, question_id: str, question: str, decomposition: str, operators: List[str] = None,
                debug: dict = None) -> TokensDependencies:
        spans_dependencies = self.spans_dependencies_extractor.extract(question_id=question_id, question=question, decomposition=decomposition, operators=operators,
                                                                       debug=debug)
        if debug is not None: debug['spans_dependencies'] = spans_dependencies.copy()
        tok_dep = self._extract(spans_dependencies, decomposition=decomposition, debug=debug)
        return tok_dep

    def _extract(self, spans_dependencies: SpansDependencies,
                 decomposition: str,
                 debug: dict = None) -> TokensDependencies:
        # collapse steps with no spans
        for collapser in self.collapsers:
            collapser.collapse(spans_dependencies, decomposition)
        if debug is not None: debug['collapsed_spans_dependencies'] = spans_dependencies

        # build tokens dependencies
        tokens_dependencies_graph = nx.MultiDiGraph()
        tokens_dependencies_graph.add_nodes_from(
            (i, {'data': TokenData(text=token.text, tag=token.bio_tag)}) for i, token in
            enumerate(spans_dependencies.tokens()))

        # update last span tokens with their corresponding step dependencies
        steps_representative_token_index = {}
        for step_id, spans_data in spans_dependencies.steps():
            if spans_data.spans_list:
                steps_representative_token_index[step_id] = spans_data.spans_list[-1].end
        for v, u, data in spans_dependencies.dependencies():
            tokens_dependencies_graph.add_edge(steps_representative_token_index[v], steps_representative_token_index[u],
                                               data=TokenDependencyData(data.dep_type))
        # update 'span' dependencies
        for _, spans_data in spans_dependencies.steps():
            tokens_indexes = list(spans_data.iter_indexes())
            for x, y in zip(tokens_indexes, tokens_indexes[1:]):
                tokens_dependencies_graph.add_edge(x, y, data=TokenDependencyData(TokenDependencyType.SPAN))

        return TokensDependencies(tokens_dependencies_graph)

    @overrides
    def get_extra_tokens(self) -> List[str]:
        return [x for c in self.collapsers for x in c.additional_tokens]

    @overrides
    def to_spans_dependencies(self, tokens_dependencies: TokensDependencies,
                              debug: dict = None) -> SpansDependencies:
        # spans dependencies graph
        spans_dependencies: SpansDependencies = super().to_spans_dependencies(tokens_dependencies, debug)
        if debug is not None: debug['pre_unwind_spans_dependencies'] = spans_dependencies.copy()

        # unwind collapsed steps
        if self.collapsers:
            for collapser in reversed(self.collapsers):
                collapser.unwind(spans_dependencies)
        return spans_dependencies


