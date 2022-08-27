from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Tuple

import networkx as nx

from qdecomp_with_dependency_graphs.scripts.qdmr_to_logical_form.operator_identifier import QDMROperator, ArgumentType
from qdecomp_with_dependency_graphs.utils.graph import render_digraph_svg


############################################
#           Data Classes                   #
############################################

DependencyType = ArgumentType

@dataclass
class StepData:
    text: str


@dataclass
class StepDependencyData:
    dep_type: DependencyType
    properties: Iterable[str] = None

    def to_string(self, include_properties: bool = True):
        return f"{self.dep_type}{'['+','.join(self.properties)+']' if (self.properties and include_properties) else ''}"

    def __str__(self):
        return self.to_string()

############################################
#         Steps Dependencies               #
############################################


class StepsView:
    def __init__(self, dep_graph: nx.MultiDiGraph):
        self._graph = dep_graph

    def __iter__(self) -> Iterable[StepData]:
        return iter(data['data'] for _, data in self._graph.nodes(data=True))

    def __getitem__(self, n) -> StepData:
        return self._graph.nodes(data=True)[n]['data']


class DependenciesView:
    def __init__(self, dep_graph: nx.MultiDiGraph, src_ids: List[int] = None):
        self._graph = dep_graph
        self._edges_view = self._graph.edges(src_ids, data=True)

    def __iter__(self) -> Iterable[Tuple[int, int, StepDependencyData]]:
        return iter((u,v,data['data']) for u, v, data in self._edges_view)


class StepsDependencies:
    def __init__(self, dependencies_graph: nx.MultiDiGraph):
        self._dependencies_graph: nx.MultiDiGraph = dependencies_graph

    def __eq__(self, other):
        return isinstance(other, StepsDependencies) and nx.algorithms.is_isomorphic(self._dependencies_graph, other._dependencies_graph)

    def steps(self) -> StepsView:
        return StepsView(self._dependencies_graph)

    def dependencies(self, src_ids: Iterable[int] = None) -> DependenciesView:
        return DependenciesView(dep_graph=self._dependencies_graph, src_ids=src_ids)

    def render_svg(self, include_properties: bool = True):
        return render_digraph_svg(self._dependencies_graph,
                                  words_text_selector=lambda id, data: id,
                                  arc_label_selector=lambda x: x['data'].to_string(include_properties=include_properties))
