from dataclasses import dataclass
from typing import Iterable, List, Tuple
import copy

import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_dependencies_graph import DependencyType
from qdecomp_with_dependency_graphs.dependencies_graph.data_types.steps_spans import Span, TokenData
from qdecomp_with_dependency_graphs.utils.graph import render_digraph_svg, reorder_by_level


@dataclass
class SpansData:
    spans_list: List[Span]

    def iter_indexes(self):
        return iter(x for span in self.spans_list for x in range(span.start, span.end+1))

    def remove_index(self, i:int):
        new_spans_list: List[Span] = []
        for s in self.spans_list:
            if i == s.start:
                s.start = i + 1
            elif i == s.end:
                s.end = i - 1
            elif s.start < i and i < s.end:
                new_spans_list.append(Span(s.start, i - 1))
                s.start = i + 1

            if s.start <= s.end:
                new_spans_list.append(s)
        self.spans_list = new_spans_list

@dataclass
class SpanDependencyData:
    dep_type: DependencyType
    properties: Iterable[str] = None

    def to_string(self, include_properties: bool = True):
        return f"{self.dep_type}{'['+','.join(self.properties)+']' if (self.properties and include_properties) else ''}"

    def __str__(self):
        return self.to_string()


class SpansView:
    def __init__(self, dep_graph: nx.MultiDiGraph):
        self._graph = dep_graph

    def __iter__(self) -> Iterable[Tuple[int, SpansData]]:
        return iter((_, data['data']) for _, data in sorted(self._graph.nodes(data=True), key=lambda x: x[0]))

    def __getitem__(self, n) -> SpansData:
        return self._graph.nodes(data=True)[n]['data']


class DependenciesView:
    def __init__(self, dep_graph: nx.MultiDiGraph, src_ids: List[int] = None):
        self._graph = dep_graph
        self._edges_view = self._graph.edges(src_ids, data=True)

    def __iter__(self) -> Iterable[Tuple[int, int, SpanDependencyData]]:
        return iter((u,v,data['data']) for u, v, data in self._edges_view)


class SpansDependencies:
    def __init__(self, dependencies_graph: nx.MultiDiGraph, tokens: Iterable[TokenData]):
        """
        :param dependencies_graph: MultiDiGraph
        where node.data['data'] is of type SpansData and edge.data['data'] is of type SpanDependencyData
        """
        self._dependencies_graph: nx.MultiDiGraph = dependencies_graph
        self._tokens: Iterable[TokenData] = list(tokens)

    def copy(self):
        return SpansDependencies(copy.deepcopy(self._dependencies_graph), list(self._tokens))

    def tokens(self) -> Iterable[TokenData]:
        return self._tokens

    def steps(self) -> SpansView:
        return SpansView(self._dependencies_graph)

    def dependencies(self, src_ids: Iterable[int] = None) -> DependenciesView:
        return DependenciesView(dep_graph=self._dependencies_graph, src_ids=src_ids)

    def render_svg(self, include_properties: bool = True):
        return render_digraph_svg(self._dependencies_graph,
                                  words_text_selector=lambda id, data: id,
                                  arc_label_selector=lambda x: x['data'].to_string(include_properties=include_properties))

    def render_html(self, level_reorder=True, include_properties: bool = True):
        if level_reorder:
            try:
                graph = self._dependencies_graph.copy()
                reorder_by_level(
                    graph,
                    key=lambda n_id, _: n_id,
                    update_node=lambda *args: None
                )
                dep_graph = SpansDependencies(dependencies_graph=graph, tokens=self._tokens)
            except Exception as ex:
                dep_graph = self
        else:
            dep_graph = self
        tokens = list(dep_graph.tokens())
        tok = '\n'.join([
            f'{i}. {"|".join(tokens[x].text for x in s.iter_indexes())}'
            for i, s in dep_graph.steps()
        ])
        deps = '\n'.join([
            f"{i} -> {v}: {dep.to_string(include_properties=include_properties)}"
            for i,_ in dep_graph.steps()
            for _, v, dep in sorted(dep_graph.dependencies(i), key=lambda x: x[1])
        ])
        return f"""
        <figure>{dep_graph.render_svg(include_properties=include_properties)}</figure>
        <pre>{tok}</pre>
        <pre>{deps}</pre>
        """

