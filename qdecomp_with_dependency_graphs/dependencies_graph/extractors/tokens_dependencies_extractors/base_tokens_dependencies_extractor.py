from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies, SpansDependencies


class BaseTokensDependenciesExtractor(ABC):
    @abstractmethod
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> TokensDependencies:
        raise NotImplementedError()

    def get_extra_tokens(self) -> List[str]:
        return []

    def to_spans_dependencies(self, tokens_dependencies: TokensDependencies,
                              debug: dict = None) -> SpansDependencies:
        # spans dependencies graph
        spans_dependencies: SpansDependencies = tokens_dependencies.to_spans_dependencies()
        return spans_dependencies
