from typing import List, Dict, Any

import networkx as nx
import spacy
from spacy.tokens import Token

import qdecomp_with_dependency_graphs.utils.data_structures as util

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, DependencyType, Span, SpanDependencyData
from qdecomp_with_dependency_graphs.dependencies_graph.evaluation.logical_form_matcher import QDMRStepTokens, QDMRStepTokensDependencies
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.operator_identifier import ArgumentType
from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.qdmr_identifier import QDMRStep, QDMROperator, get_identifier


class SpansDepToQDMRStepTokensConverter:
    def __init__(self, infer_properties: bool = True):
        super().__init__()
        self.infer_properties = infer_properties

    def convert(self, spans_dependencies: SpansDependencies, question_id: str = None) -> QDMRStepTokensDependencies:
        """
        Convert spans dependencies to a graph where the spans nodes are replaces with QDMRStepTokens
        The edges are not changed
        :param spans_dependencies:
        :param question_id:
        :return:
        """
        question_tokens = [x.text for x in spans_dependencies.tokens()]
        steps_graph: nx.MultiDiGraph = spans_dependencies._dependencies_graph.copy()

        for n, data in steps_graph.nodes(data=True):
            spans: SpansData = data.get('data', None)
            spans_list = spans.spans_list if spans else []
            outgoing_edges = steps_graph.out_edges(n, data=True)

            tokens = [question_tokens[i] for span in spans_list for i in range(span.start, span.end+1)]
            step = self._get_step(outgoing_edges, tokens)
            data['data'] = step

        return QDMRStepTokensDependencies(dependencies_graph=steps_graph)

    def _get_step(self, outgoing_edges, tokens: List[str]) -> QDMRStepTokens:
        dependencies: List[SpanDependencyData] = [data['data'] for _,_,data in outgoing_edges]
        refs_types: Dict[DependencyType, Any] = util.list_to_multivalue_dict(outgoing_edges, key=lambda x: x[2]['data'].dep_type)

        # select
        if not refs_types:
            return QDMRStepTokens(operator=QDMROperator.SELECT,
                                  properties=None,
                                  arguments_tokens={ArgumentType.SELECT_SUB: tokens})

        operators = set(x.get_operator() for x in refs_types)
        if len(operators) != 1:
            raise ValueError(f"could not detect a single operator among {operators} ({outgoing_edges})")

        operator = list(operators)[0]
        properties = dependencies[0].properties  # todo: verify that all the dependencies have the same properties
        if self.infer_properties and not properties:
            identifier = get_identifier(operator)
            refs = [v for u, v, *_ in outgoing_edges]
            properties = identifier.extract_properties(' '.join(tokens), references=refs)
            if not properties:
                properties = self.get_default_properties(operator, list(refs_types.keys()), refs, tokens)


        # clean tokens based on operator
        # todo: is necessary? it should be sync with the spans alignment logic
        if operator in [QDMROperator.AGGREGATE, QDMROperator.ARITHMETIC]:
            if ' '.join(tokens[:2]).lower() in ['how many', 'how much']:
                tokens = tokens[2:]

        single_option = [
            DependencyType.BOOLEAN_CONDITION, DependencyType.COMPARATIVE_CONDITION, DependencyType.FILTER_CONDITION,
            DependencyType.INTERSECTION_PROJECTION, DependencyType.PROJECT_PROJECTION
        ]
        for x in single_option:
            if x.get_operator() == operator:
                return QDMRStepTokens(operator=operator, properties=properties,
                                      arguments_tokens={x: tokens})

        # set the tokens for the arg group with no references, or to the first one
        # example: difference of 100 and #1 => arithmetic-left has no references, so it will get the "100"
        multi_options = [
            (DependencyType.ARITHMETIC_LEFT, DependencyType.ARITHMETIC_RIGHT),
            (DependencyType.DISCARD_SUB, DependencyType.DISCARD_EXCLUDE),
            (DependencyType.GROUP_VALUE, DependencyType.GROUP_KEY),
            (DependencyType.SORT_SUB, DependencyType.SORT_ORDER),
        ]
        for x, y in multi_options:
            if x in refs_types or y in refs_types:
                arg_name = x if y in refs_types else y
                return QDMRStepTokens(operator=operator, properties=properties,
                                      arguments_tokens={arg_name: tokens})

        return QDMRStepTokens(operator=operator, properties=properties, arguments_tokens={})

    @staticmethod
    def get_default_properties(operator: QDMROperator, references_type: List[DependencyType],
                               references: List[int] = None, tokens: List[Token] = None):
        # todo: move it to each "extract operator"?
        # defaults:
        #   arithmetic: sum?
        #   boolean (left, right) : equals (just if tokens empty? after cleaning...)
        #   boolean (arg): any? is true?
        #   group: count
        #   superlative: highest

        identifier = get_identifier(operator)
        prop = {
            QDMROperator.AGGREGATE: identifier.extract_properties("number of", references),
            QDMROperator.COMPARISON: identifier.extract_properties("true", references),
            QDMROperator.GROUP: identifier.extract_properties("number of", references),
            QDMROperator.SUPERLATIVE: identifier.extract_properties("highest", references)
        }.get(operator, None)
        if prop is not None:
            return prop

        if operator == QDMROperator.BOOLEAN and not tokens:
            if DependencyType.BOOLEAN_CONDITION in references_type:
                return identifier.extract_properties("equals to", references)
            return identifier.extract_properties("true", references)

        if operator == QDMROperator.ARITHMETIC:
            if any(x in references_type for x in [DependencyType.ARITHMETIC_LEFT, DependencyType.ARITHMETIC_RIGHT]):
                return identifier.extract_properties("difference", references)
            return identifier.extract_properties("sum", references)