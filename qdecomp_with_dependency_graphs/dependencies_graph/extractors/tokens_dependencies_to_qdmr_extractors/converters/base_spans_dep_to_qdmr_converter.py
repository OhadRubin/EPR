from abc import ABC, abstractmethod

import networkx as nx

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies
from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition


class BaseSpansDepToQdmrConverter(ABC):
    @abstractmethod
    def convert(self, spans_dependencies: SpansDependencies) -> Decomposition:
        raise NotImplementedError()
