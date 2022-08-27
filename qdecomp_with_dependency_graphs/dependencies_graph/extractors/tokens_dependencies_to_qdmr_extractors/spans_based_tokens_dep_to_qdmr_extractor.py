from typing import List

from overrides import overrides


from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, TokensDependencies
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_extractors import BaseTokensDependenciesExtractor
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_to_qdmr_extractors.converters import BaseSpansDepToQdmrConverter
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer

from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_to_qdmr_extractors.base_tokens_dep_to_qdmr_extractor import \
    BaseTokensDependenciesToQDMRExtractor


class SpansBasedTokensDependenciesToQDMRExtractor(BaseTokensDependenciesToQDMRExtractor):
    def __init__(self, converter: BaseSpansDepToQdmrConverter,
                 tokens_dependencies_extractor: BaseTokensDependenciesExtractor):
        self.converter = converter
        self.tokens_dependencies_extractor = tokens_dependencies_extractor

    @overrides
    def extract(self, tokens_dependencies: TokensDependencies, debug: dict = None) -> Decomposition:
        spans_dependencies: SpansDependencies = self.tokens_dependencies_extractor.to_spans_dependencies(
            tokens_dependencies=tokens_dependencies, debug=debug)
        if debug is not None: debug['spans_dependencies'] = spans_dependencies.copy()

        # convert to qdmr
        decomposition = self.converter.convert(spans_dependencies)
        qdmr_graph = decomposition.to_graph()

        # reorder (reindex steps ids, remove duplicates, ...)
        NormalizedGraphMatchScorer.reorder(qdmr_graph)

        # to decomposition
        return Decomposition.from_graph(qdmr_graph)
