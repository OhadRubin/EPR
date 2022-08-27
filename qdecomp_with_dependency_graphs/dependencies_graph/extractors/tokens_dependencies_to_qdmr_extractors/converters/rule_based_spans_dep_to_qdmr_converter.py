from collections import Counter
from typing import List, Dict, Tuple, Any

import networkx as nx
from overrides import overrides

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, SpansData, DependencyType
from qdecomp_with_dependency_graphs.dependencies_graph.extractors.tokens_dependencies_to_qdmr_extractors.converters import BaseSpansDepToQdmrConverter
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.utils import data_structures as util


class RuleBasedSpansDepToQdmrConverter(BaseSpansDepToQdmrConverter):
    @overrides
    def convert(self, spans_dependencies: SpansDependencies) -> Decomposition:
        graph = _convert_dependencies_to_qdmr(spans_dependencies)
        return Decomposition.from_graph(graph)


def _convert_dependencies_to_qdmr(spans_dependencies: SpansDependencies) -> nx.DiGraph:
    question_tokens = [x.text for x in spans_dependencies.tokens()]
    dependencies_graph = spans_dependencies._dependencies_graph
    decomposition_graph = nx.DiGraph()

    for n, data in dependencies_graph.nodes(data=True):
        spans: SpansData = data.get('data', None)
        spans_list = spans.spans_list if spans else []
        spans_str = [' '.join(str(x) for x in question_tokens[s.start:s.end+1]) for s in spans_list]
        outgoing_edges = dependencies_graph.out_edges(n, data=True)

        label, refs = _get_label_and_refs(outgoing_edges, spans_str)
        if label is None:
            raise ValueError(f'could not build qdmr for step {spans_str}, {outgoing_edges}')
        if refs:
            decomposition_graph.add_edges_from((n,x) for x in refs)
        if n not in decomposition_graph:
            decomposition_graph.add_node(n)
        decomposition_graph.nodes[n]['label'] = label
    return decomposition_graph


def _get_refs_if_match(refs_types: Dict[
    DependencyType, List[Tuple[int, int, Dict[str, Any]]]],
                       dep_sequence: List[
                           DependencyType],
                       strict_count=True):
    def node_to_ref(node_id: int):
        return f'@@{node_id}@@'

    dep_counts = Counter(dep_sequence)
    valid = all(x in refs_types for x in dep_counts) and len(dep_counts) == len(refs_types) \
           and (not strict_count or all(len(refs_types[x]) == dep_counts[x] for x in dep_sequence))
    if not valid:
        return None
    next_index = {x:0 for x in refs_types}
    refs = []
    for x in dep_sequence:
        if strict_count:
            refs.append(refs_types[x][next_index[x]])
            next_index[x] += 1
        else:
            refs.extend(refs_types[x])
    refs_ids = [v for u,v,*_ in refs]
    return refs_ids, [node_to_ref(x) for x in refs_ids]


def _get_label_and_refs(outgoing_edges, ranges_str:List[str]):
    tokens_str = ' '.join(ranges_str)
    refs_types = util.list_to_multivalue_dict(outgoing_edges, key=lambda x: x[2]['data'].dep_type)

    # select
    if not refs_types:
        return tokens_str, None

    # aggregate
    res = _get_refs_if_match(refs_types, [
        DependencyType.AGGREGATE_ARG])
    if res:
        refs, refs_tokens = res
        aggregate_arg, = refs_tokens
        agg = tokens_str if tokens_str else 'number of '
        return f'{agg} {aggregate_arg}', refs

    # arithmetic
    res = _get_refs_if_match(refs_types, [
        DependencyType.ARITHMETIC_LEFT, DependencyType.ARITHMETIC_RIGHT])
    if res:
        refs, refs_tokens = res
        arithmetic_left, arithmetic_right = refs_tokens
        if len(ranges_str) == 1:
            return f'{arithmetic_left} {ranges_str[0]} {arithmetic_right}', refs
        return f'{ranges_str[0]} {arithmetic_left} {" ".join(ranges_str[1:])} {arithmetic_right}', refs

    # boolean
    res = _get_refs_if_match(refs_types, [
        DependencyType.BOOLEAN_SUB, DependencyType.BOOLEAN_CONDITION])
    if res:
        refs, refs_tokens = res
        boolean_left, boolean_right = refs_tokens
        if len(ranges_str) == 1:
            return f'{boolean_left} {ranges_str[0]} {boolean_right}', refs
        return f'{ranges_str[0]} {boolean_left} {" ".join(ranges_str[1:]) or "equals"} {boolean_right}', refs

    res = _get_refs_if_match(refs_types, [
        DependencyType.BOOLEAN_SUB])
    if res:
        refs, refs_tokens = res
        boolean_arg, = refs_tokens
        return f'if {boolean_arg} {tokens_str}', refs

    # comparative
    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE])
    if res:
        refs, refs_tokens = res
        comparative_sub, comparative_arg = refs_tokens
        return f'{comparative_sub} if {comparative_arg} {tokens_str}', refs

    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARATIVE_SUB, DependencyType.COMPARATIVE_ATTRIBUTE, DependencyType.COMPARATIVE_CONDITION])
    if res:
        refs, refs_tokens = res
        comparative_sub, comparative_left, comparative_right = refs_tokens
        return f'{comparative_sub} where {comparative_left} {tokens_str} {comparative_right}', refs

    # comparison
    res = _get_refs_if_match(refs_types, [
        DependencyType.COMPARISON_ARG])
    if res:
        refs, refs_tokens = res
        comparison_arg, = refs_tokens
        return f'if {comparison_arg} {tokens_str}', refs

    # discard
    res = _get_refs_if_match(refs_types, [
        DependencyType.DISCARD_SUB, DependencyType.DISCARD_EXCLUDE])
    if res:
        refs, refs_tokens = res
        discard_sub, discard_arg = refs_tokens
        return f'{discard_sub} besides {discard_arg}', refs

    # filter
    res = _get_refs_if_match(refs_types, [
        DependencyType.FILTER_SUB])
    if res:
        refs, refs_tokens = res
        filter_sub, = refs_tokens
        return f'{filter_sub} {tokens_str}', refs

    res = _get_refs_if_match(refs_types, [
        DependencyType.FILTER_SUB, DependencyType.FILTER_CONDITION])
    if res:
        refs, refs_tokens = res
        filter_sub, filter_arg = refs_tokens
        return f'{filter_sub} {tokens_str} {filter_arg}', refs

    # group
    res = _get_refs_if_match(refs_types, [
        DependencyType.GROUP_VALUE, DependencyType.GROUP_KEY])
    if res:
        refs, refs_tokens = res
        group_arg, group_by = refs_tokens
        return f'number of {group_arg} for each {group_by}', refs

    # intersection
    res = _get_refs_if_match(refs_types, [
        DependencyType.INTERSECTION_PROJECTION, DependencyType.INTERSECTION_INTERSECTION, DependencyType.INTERSECTION_INTERSECTION])
    if res:
        refs, refs_tokens = res
        intersection_sub, intersection_arg, intersection_arg = refs_tokens
        return f'{intersection_sub} in both {intersection_arg} and {intersection_arg}', refs

    # project
    res = _get_refs_if_match(refs_types, [
        DependencyType.PROJECT_SUB])
    if res:
        refs, refs_tokens = res
        project_arg, = refs_tokens
        return f'{tokens_str} of {project_arg}', refs

    res = _get_refs_if_match(refs_types, [
        DependencyType.PROJECT_PROJECTION, DependencyType.PROJECT_SUB])
    if res:
        refs, refs_tokens = res
        project_sub, project_arg = refs_tokens
        return f'{project_sub} of {project_arg}', refs

    # sort
    res = _get_refs_if_match(refs_types, [
        DependencyType.SORT_SUB, DependencyType.SORT_ORDER])
    if res:
        refs, refs_tokens = res
        sort_sub, sort_arg = refs_tokens
        return f'{sort_sub} sorted by {sort_arg}', refs

    # superlative
    res = _get_refs_if_match(refs_types, [
        DependencyType.SUPERLATIVE_SUB, DependencyType.SUPERLATIVE_ATTRIBUTE])
    if res:
        refs, refs_tokens = res
        superlative_sub, superlative_arg = refs_tokens
        return f'{superlative_sub} where {superlative_arg} {tokens_str}', refs

    # union
    res = _get_refs_if_match(refs_types, [
        DependencyType.UNION_SUB], strict_count=False)
    if res:
        refs, refs_tokens = res
        return ','.join(refs_tokens), refs

    return None, None