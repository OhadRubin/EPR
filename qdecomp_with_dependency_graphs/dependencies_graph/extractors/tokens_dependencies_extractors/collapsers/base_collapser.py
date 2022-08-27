from abc import ABC, abstractmethod
from typing import List, Tuple

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies, DependencyType


class BaseCollapser(ABC):
    def __init__(self):
        self.additional_tokens = []

    """
    Deal with empty spans in SpansDependencies graph
    """
    @abstractmethod
    def collapse(self, spans_dependencies: SpansDependencies, decomposition: str) -> None:
        raise NotImplementedError()

    @abstractmethod
    def unwind(self, spans_dependencies: SpansDependencies) -> None:
        raise NotImplementedError()

    @staticmethod
    def _get_operator(x: str):
        return DependencyType(x).get_operator() if DependencyType.has_value(x) else x