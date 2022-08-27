from typing import List, Tuple

import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, StepsSpans, StepsDependencies, SpansData
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.spans_dependencies_graph import SpanDependencyData
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.spans_dependencies_extractors import BaseSpansDependenciesExtractor
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_dependencies_extractors import BaseStepsDependenciesExtractor
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.steps_spans_extractors import BaseSpansExtractor


class MergeSpansDependenciesExtractor(BaseSpansDependenciesExtractor):
    def __init__(self, spans_extractor: BaseSpansExtractor, steps_dependencies_extractor: BaseStepsDependenciesExtractor):
        self.spans_extractor = spans_extractor
        self.steps_dependencies_extractor = steps_dependencies_extractor

    def extract(self, question_id: str, question: str, decomposition: str, operators: List[str] = None,
                debug: dict = None) -> SpansDependencies:
        steps_spans = self.spans_extractor.extract(question_id=question_id, question=question,
                                                      decomposition=decomposition, operators=operators,
                                                   debug = debug)
        if debug is not None: debug['steps_spans'] = steps_spans.copy()
        steps_dependencies = self.steps_dependencies_extractor.extract(question_id=question_id, question=question,
                                                                          decomposition=decomposition,
                                                                          operators=operators,
                                                                       debug=debug)

        if debug is not None: debug['steps_dependencies'] = steps_dependencies
        return self._merge_spans_and_steps_dependencies(steps_spans, steps_dependencies)


    @staticmethod
    def _merge_spans_and_steps_dependencies(steps_spans: StepsSpans,
                                      steps_dependencies: StepsDependencies) -> SpansDependencies:
        dependencies_graph = nx.MultiDiGraph()
        # update spans data
        for i, spans in steps_spans:
            dependencies_graph.add_node(i, data=SpansData(spans))

        # update dependencies
        for u, v, data in steps_dependencies.dependencies():
            dependencies_graph.add_edge(u, v, data=SpanDependencyData(data.dep_type, properties=data.properties))

        return SpansDependencies(dependencies_graph=dependencies_graph, tokens=steps_spans.tokens())