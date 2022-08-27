from abc import abstractmethod, ABC
from typing import List, Tuple

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import SpansDependencies


class BaseSpansDependenciesExtractor(ABC):
    @abstractmethod
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> SpansDependencies:
        raise NotImplementedError()