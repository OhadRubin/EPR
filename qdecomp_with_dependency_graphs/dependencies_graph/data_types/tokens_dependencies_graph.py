from dataclasses import dataclass
from enum import Enum
from typing import Union, List, Iterable, Tuple

import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types.spans_dependencies_graph import Span, DependencyType, SpansData, SpansDependencies, \
    SpanDependencyData
from qdecomp_with_dependency_graphs.utils import data_structures as util
from qdecomp_with_dependency_graphs.utils.graph import render_digraph_svg

#################################
#       Data Classes            #
#################################

class TokenDependencyType(str, Enum):
    SPAN = 'span'
    NONE = 'none'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    def __str__(self):
        return self.value


@dataclass
class TokenData:
    text: str
    tag: str


@dataclass
class TokenDependencyData:
    dep_type: Union[DependencyType, TokenDependencyType]


#################################
#     Tokens Dependencies       #
#################################

class TokensView:
    def __init__(self, tokens_dep_graph: nx.MultiDiGraph):
        self._graph = tokens_dep_graph

    def __iter__(self) -> Iterable[TokenData]:
        return iter(data['data'] for _, data in self._graph.nodes(data=True))

    def __getitem__(self, n) -> TokenData:
        return self._graph.nodes(data=True)[n]['data']


class DependenciesView:
    def __init__(self, tokens_dep_graph: nx.MultiDiGraph, tokens_indexes: List[int] = None):
        self._graph = tokens_dep_graph
        self._edges_view = self._graph.edges(tokens_indexes, data=True)

    def __iter__(self) -> Iterable[Tuple[int, int, TokenDependencyData]]:
        return iter((u,v,data['data']) for u, v, data in self._edges_view)

class TokenDependenciesView:
    def __init__(self, tokens_dep_graph: nx.MultiDiGraph, is_in: bool = False, tokens_indexes: List[int] = None):
        self._graph = tokens_dep_graph
        self._tokens_indexes = tokens_indexes or self._graph.nodes
        self._is_in = is_in
        self._edges_view = self._graph.in_edges(tokens_indexes, data=True) if is_in else self._graph.edges(tokens_indexes, data=True)

    def _get_tokens_deps(self):
        return util.list_to_multivalue_dict([(u, v, data['data']) for u, v, data in self._edges_view],
                                            key=lambda x: x[1 if self._is_in else 0])

    def __iter__(self) -> Iterable[Tuple[int, Iterable[Tuple[int, int, TokenDependencyData]]]]:
        return iter(self._get_tokens_deps().items())

    def __getitem__(self, item) -> Iterable[Tuple[int, int, TokenDependencyData]]:
        return self._get_tokens_deps()[item]


class TokensDependencies:
    def __init__(self, dependencies_graph: nx.MultiDiGraph):
        self._dependencies_graph: nx.MultiDiGraph = dependencies_graph
        self._tokens = TokensView(tokens_dep_graph=self._dependencies_graph)

    @staticmethod
    def from_tokens(tokens: List[str], token_tags: List[str], dependencies: List[Tuple[int, int, str]]) -> 'TokensDependencies':
        assert len(tokens) == len(token_tags), f'mismatched tokens and tags amount {len(tokens), len(token_tags)}'
        def str_to_dependency(dep_str: str) -> Union[DependencyType, TokenDependencyType]:
            if TokenDependencyType.has_value(dep_str):
                return TokenDependencyType(dep_str)
            if DependencyType.has_value(dep_str):
                return DependencyType(dep_str)
            #raise ValueError(f"Unrecognized dependency: {dep_str}")
            return dep_str  # used for collapser that create str instead of proper dependency
        dep_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        dep_graph.add_nodes_from([(i, dict(data=TokenData(x, y))) for i, (x, y) in enumerate(zip(tokens, token_tags))])
        dep_graph.add_edges_from([(i, j, dict(data=TokenDependencyData(str_to_dependency(t)))) for i, j, t in dependencies])
        return TokensDependencies(dep_graph)

    def tokens(self) -> TokensView:
        return self._tokens

    def dependencies(self, src_tokens_indexes: Iterable[int] = None) -> DependenciesView:
        return DependenciesView(tokens_dep_graph=self._dependencies_graph, tokens_indexes=src_tokens_indexes)

    def tokens_dependencies(self, tokens_indexes: Iterable[int] = None, in_dependencies: bool = False) -> TokenDependenciesView:
        return TokenDependenciesView(tokens_dep_graph=self._dependencies_graph, tokens_indexes=tokens_indexes,
                                     is_in=in_dependencies)

    def render_svg(self):
        return render_digraph_svg(self._dependencies_graph,
                                  words_text_selector=lambda id, data: data['data'].text,
                                  words_tag_selector=lambda id, data: data['data'].tag,
                                  arc_label_selector=lambda x: x['data'].dep_type)

    def to_spans_dependencies(self) -> SpansDependencies:
        decomp_graph = nx.MultiDiGraph()

        # map spans to nodes
        token_index_to_step = {}
        steps_next_id = 1
        spans_edges = ((u,v,dep) for u,v,dep in self.dependencies() if dep.dep_type == TokenDependencyType.SPAN)
        for u,v, dep in spans_edges:
            u_step = token_index_to_step.get(u, None)
            v_step = token_index_to_step.get(v, None)
            if u_step is None and v_step is None:
                token_index_to_step[u] = steps_next_id
                token_index_to_step[v] = steps_next_id
                steps_next_id += 1
            elif u_step is None:
                token_index_to_step[u] = v_step
            elif v_step is None:
                token_index_to_step[v] = u_step
            elif v_step != u_step:
                src_step = max(u_step, v_step)
                dst_step = min(u_step, v_step)
                for k in token_index_to_step:
                    if token_index_to_step[k] == src_step:
                        token_index_to_step[k] = dst_step
        for n in self._dependencies_graph:  # singletons spans
            # todo: deal with singleton with no incoming dependency (eg: ATIS_dev_121)
            if n not in token_index_to_step and self._dependencies_graph.degree(n) > 0:
                token_index_to_step[n] = steps_next_id
                steps_next_id += 1
        steps_to_token_index = util.swap_keys_and_values_dict(token_index_to_step)
        for step_id, tokens_indexes in steps_to_token_index.items():
            step_spans = [Span(start, end) for start, end in util.find_ranges(tokens_indexes)]
            decomp_graph.add_node(step_id, data=SpansData(step_spans))

        # add dependencies
        dependencies = ((u,v,SpanDependencyData(dep_type=d.dep_type)) for u,v,d in self.dependencies() if d.dep_type not in [TokenDependencyType.SPAN, TokenDependencyType.NONE])
        decomp_graph.add_edges_from((token_index_to_step[u],
                                     token_index_to_step[v],
                                     {'data': dep})
                                    for u,v,dep in dependencies)

        return SpansDependencies(decomp_graph, self.tokens())