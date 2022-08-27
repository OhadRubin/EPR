from abc import ABC, abstractmethod
from typing import List, Tuple

from qdecomp_with_dependency_graphs.evaluation.decomposition import Decomposition
from qdecomp_with_dependency_graphs.dependencies_graph.data_types import TokensDependencies


class BaseTokensDependenciesToQDMRExtractor(ABC):
    @abstractmethod
    def extract(self, tokens_dependencies: TokensDependencies, debug: dict = None) -> Decomposition:
        raise NotImplementedError()